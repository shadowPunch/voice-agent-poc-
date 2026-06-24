import asyncio
import json
import uuid
import re
import fractions
import time
import sqlite3
import random
import os
import collections
import concurrent.futures
import numpy as np
import torch
import av
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import uvicorn
from faster_whisper import WhisperModel
from kokoro import KPipeline

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

app = FastAPI()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INITIALIZATION: DB, RAG, & MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("Initializing Models and Databases...")

# Load Mistral API key from environment variable, falling back to a token file if present
if "MISTRAL_API_KEY" not in os.environ:
    token_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mistral_token.txt")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            os.environ["MISTRAL_API_KEY"] = f.read().strip()

if "MISTRAL_API_KEY" not in os.environ or not os.environ["MISTRAL_API_KEY"]:
    print("WARNING: MISTRAL_API_KEY environment variable is not set. The agent will fail to run inference without it.")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Increased workers to handle concurrent LLM and TTS tasks
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# --- SQLite Setup ---
DB_PATH = "inventory.db"
EXCEL_PATH = "inventory_report.xlsx"
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS inventory (item_name TEXT PRIMARY KEY, quantity INTEGER)")
    cursor.execute("SELECT COUNT(*) FROM inventory")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO inventory (item_name, quantity) VALUES (?, ?)", 
                           [("apples", 50), ("bananas", 20), ("waterbottles", 100)])
        conn.commit()
    conn.close()
init_db()

# --- ChromaDB Setup ---
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "knowledge"
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _chroma_col = _chroma_client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "l2"})
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# --- AI Models ---
stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", verbose=False, trust_repo=True)
_, _, _, VADIterator, _ = vad_utils

# --- Contextual Fillers ---
FILLERS = {
    "database": ["Let me check the records.", "Accessing the database now.", "Working on that now."],
    "knowledge": ["Hmm, let me look that up.", "Let me search the knowledge base."],
    "general": ["Just a moment.", "Let me think about that.", "Give me a second."],
    "short": ["Hmm.", "Okay.", "Right."]
}
_last_filler_used = None
_filler_pcm = {}
AUDIO_PAD_SECONDS = 0.5

for cat, phrases in FILLERS.items():
    for phrase in phrases:
        parts = [a for _, _, a in tts_pipeline(phrase, voice="af_sky")]
        if parts:
            pcm = (torch.cat(parts).numpy() * 32767).astype(np.int16)
            pad = np.zeros(int(24000 * AUDIO_PAD_SECONDS), dtype=np.int16) 
            _filler_pcm[phrase] = np.concatenate([pad, pcm])

def _pick_filler(transcript: str) -> str:
    global _last_filler_used
    t = transcript.lower()
    if len(t.split()) < 4 and any(w in t for w in ["hi", "hello", "hey", "thanks", "ok", "got it"]): return None
    if any(w in t for w in ["inventory", "database", "excel", "update", "export"]): cat = "database"
    elif any(w in t for w in ["what", "how", "why", "who", "when"]): cat = "knowledge"
    elif len(t.split()) < 5: cat = "short"
    else: cat = "general"
    opts = [f for f in FILLERS[cat] if f != _last_filler_used] or FILLERS[cat]
    chosen = random.choice(opts)
    _last_filler_used = chosen
    return chosen

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LANGGRAPH BRAIN
# ═══════════════════════════════════════════════════════════════════════════════
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for general knowledge or documents."""
    if not RAG_AVAILABLE or _chroma_col.count() == 0: return "Knowledge base empty."
    emb = _embed_model.encode([query], convert_to_numpy=True)
    res = _chroma_col.query(query_embeddings=emb.tolist(), n_results=3, include=["documents", "distances"])
    kept = [d for d, dist in zip(res["documents"][0], res["distances"][0]) if dist < 1.2]
    return "Context:\n" + "\n---\n".join(kept) if kept else "No info found."

@tool
def check_inventory(item_name: str = None) -> str:
    """Check quantity of a specific item, or all items."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    if item_name:
        cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (item_name.lower(),))
        res = cursor.fetchone()
        conn.close()
        return f"We have {res[0]} {item_name}." if res else f"Item '{item_name}' not found."
    cursor.execute("SELECT item_name, quantity FROM inventory")
    res = cursor.fetchall()
    conn.close()
    return "Current inventory: " + ", ".join([f"{n}: {q}" for n, q in res]) if res else "Inventory empty."

@tool
def update_inventory(item_name: str, quantity: int) -> str:
    """Update quantity of a specific item."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    item = item_name.lower()
    cursor.execute("SELECT item_name FROM inventory WHERE item_name = ?", (item,))
    if cursor.fetchone(): cursor.execute("UPDATE inventory SET quantity = ? WHERE item_name = ?", (quantity, item))
    else: cursor.execute("INSERT INTO inventory (item_name, quantity) VALUES (?, ?)", (item, quantity))
    conn.commit()
    conn.close()
    return f"Updated {item} to {quantity}."

@tool
def export_to_excel() -> str:
    """Export the current inventory database to an Excel (.xlsx) file."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    df = pd.read_sql_query("SELECT * FROM inventory", conn)
    conn.close()
    df.to_excel(EXCEL_PATH, index=False)
    return f"Inventory exported to {EXCEL_PATH}."

tools = [search_knowledge_base, check_inventory, update_inventory, export_to_excel]
llm = ChatMistralAI(model="ministral-8b-latest", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    msgs = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in msgs):
        sys_prompt = (
            "You are Alex, a friendly and casual human colleague. "
            "Speak naturally using contractions (like 'I'm', 'let's', 'we've'). "
            "Never act like an AI, an assistant, or a robot. "
            "Keep your answers to 1 or 2 short, spoken sentences. "
            "Never use markdown, asterisks, or lists."
        )
        msgs = [SystemMessage(content=sys_prompt)] + msgs
    return {"messages": [llm_with_tools.invoke(msgs)]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
graph = workflow.compile(checkpointer=MemorySaver())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. WEBSOCKET UI MANAGER & OBSERVABILITY TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════
class UIManager:
    def __init__(self):
        self.rooms = collections.defaultdict(list)
        
    async def connect(self, ws: WebSocket, room_id: str):
        await ws.accept()
        self.rooms[room_id].append(ws)
        
    def disconnect(self, ws: WebSocket, room_id: str):
        if room_id in self.rooms and ws in self.rooms[room_id]: 
            self.rooms[room_id].remove(ws)
            if not self.rooms[room_id]: del self.rooms[room_id]
                
    async def broadcast(self, data: dict, room_id: str):
        if room_id in self.rooms:
            for conn in self.rooms[room_id]:
                try: await conn.send_json(data)
                except: pass

ui_manager = UIManager()

class TurnTelemetry:
    def __init__(self, room_id, user_delay_sec, audio_peak, audio_duration_sec):
        self.room_id = room_id
        self.audio_duration = audio_duration_sec
        self.timestamps = {"turn_start": time.time()}
        self.metrics = {
            "user_delay": f"{user_delay_sec:.1f}s",
            "audio_quality": f"{audio_peak:.2f} Peak (RMS)",
            "eos_latency": "500 ms"
        }

    def mark(self, event_name):
        # Prevent overwriting if already marked (e.g. for first sentence vs full)
        if event_name not in self.timestamps:
            self.timestamps[event_name] = time.time()

    async def flush(self):
        ts = self.timestamps
        if "stt_start" in ts and "stt_end" in ts:
            self.metrics["stt_latency"] = f"{(ts['stt_end'] - ts['stt_start'])*1000:.0f} ms"
        if "llm_start" in ts and "llm_ttft" in ts:
            self.metrics["llm_ttft"] = f"{(ts['llm_ttft'] - ts['llm_start'])*1000:.0f} ms"
        if "tts_start" in ts and "tts_first_audio" in ts:
            self.metrics["tts_latency"] = f"{(ts['tts_first_audio'] - ts['tts_start'])*1000:.0f} ms"
        
        if "tts_first_audio" in ts:
            ttfa = ts['tts_first_audio'] - ts['turn_start']
            self.metrics["first_audio"] = f"{ttfa*1000:.0f} ms"
            self.metrics["total_rt"] = f"{(ttfa + 0.5)*1000:.0f} ms"
            rtf = ttfa / self.audio_duration if self.audio_duration > 0 else 0
            self.metrics["rtf"] = f"{rtf:.2f}x"
            
        await ui_manager.broadcast({"type": "telemetry", "data": self.metrics}, self.room_id)

@app.websocket("/ws/ui/{room_id}")
async def ui_endpoint(websocket: WebSocket, room_id: str):
    await ui_manager.connect(websocket, room_id)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong", "time": msg.get("time")})
    except WebSocketDisconnect:
        ui_manager.disconnect(websocket, room_id)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. WEBRTC AUDIO PIPELINE WITH STREAMING TTS
# ═══════════════════════════════════════════════════════════════════════════════
class AgentAudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.sample_rate = 24000
        self.samples_per_frame = int(self.sample_rate * 0.02)
        self._timestamp = 0
        self._start_time = None
        self.is_speaking = False
        self.stop_requested = False
        self.last_speech_end = time.time() 

    def clear_queue(self):
        self.stop_requested = True
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except: pass
        self.is_speaking = False

    async def add_tts_audio(self, pcm_data: np.ndarray):
        if self.stop_requested: return
        self.is_speaking = True
        for i in range(0, len(pcm_data), self.samples_per_frame):
            if self.stop_requested: break
            chunk = pcm_data[i:i+self.samples_per_frame]
            if len(chunk) < self.samples_per_frame:
                chunk = np.pad(chunk, (0, self.samples_per_frame - len(chunk)))
            await self.queue.put(chunk)

    async def recv(self):
        try: chunk = await asyncio.wait_for(self.queue.get(), timeout=0.02)
        except asyncio.TimeoutError: chunk = np.zeros(self.samples_per_frame, dtype=np.int16)
        
        frame = av.AudioFrame(format='s16', layout='mono', samples=self.samples_per_frame)
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(chunk.tobytes())
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += self.samples_per_frame
        
        if self._start_time is None:
            self._start_time = time.time()
        else:
            wait_time = (self._timestamp / self.sample_rate) - (time.time() - self._start_time)
            if wait_time > 0: await asyncio.sleep(wait_time)
            
        return frame

def _run_stt(audio_data):
    # OPTIMIZATION: Disabled vad_filter. Silero already VAD'd the audio perfectly.
    segs, _ = stt_model.transcribe(audio_data, beam_size=1, temperature=0.0, language="en", vad_filter=False)
    return " ".join([s.text.strip() for s in segs]).strip()

def _run_tts(text):
    return [a for _, _, a in tts_pipeline(text, voice="af_sky")]

async def process_turn(audio_data: np.ndarray, session_id: str, agent_track: AgentAudioTrack, room_id: str, telemetry: TurnTelemetry):
    agent_track.stop_requested = False
    await ui_manager.broadcast({"type": "state", "val": "Processing..."}, room_id)
    loop = asyncio.get_running_loop()
    
    try:
        # 1. STT Phase
        telemetry.mark("stt_start")
        text = await loop.run_in_executor(_executor, _run_stt, audio_data)
        telemetry.mark("stt_end")
        
        words = [w.lower().strip(".,!?") for w in text.split()]
        if not text or len(text) < 2 or (len(words) > 3 and len(set(words)) == 1): 
            return await ui_manager.broadcast({"type": "state", "val": "Listening..."}, room_id)
            
        await ui_manager.broadcast({"type": "user", "text": text}, room_id)
        
        # 2. Filler Logic (OPTIMIZED: No sleep, unblocks LLM immediately)
        filler_text = _pick_filler(text)
        if filler_text and not agent_track.stop_requested:
            await agent_track.add_tts_audio(_filler_pcm[filler_text])
        
        # 3. Pipelined LangGraph -> TTS Phase
        if agent_track.stop_requested: return
        telemetry.mark("llm_start")
        await ui_manager.broadcast({"type": "agent_start", "val": "Thinking..."}, room_id)
        
        # Async Queue to handle sentence chunks
        tts_queue = asyncio.Queue()
        
        async def tts_consumer():
            """Runs concurrently with LangGraph to synthesize sentences instantly"""
            first_sentence_done = False
            while True:
                sentence = await tts_queue.get()
                if sentence is None: break # EOF signal
                if agent_track.stop_requested: continue
                
                parts = await loop.run_in_executor(_executor, _run_tts, sentence)
                if parts and not agent_track.stop_requested:
                    pcm = (torch.cat(parts).numpy() * 32767).astype(np.int16)
                    
                    if not first_sentence_done:
                        telemetry.mark("tts_first_audio")
                        pad = np.zeros(int(24000 * AUDIO_PAD_SECONDS), dtype=np.int16) 
                        pcm = np.concatenate([pad, pcm])
                        first_sentence_done = True
                        
                    await agent_track.add_tts_audio(pcm)

        # Start the TTS worker in the background
        consumer_task = asyncio.create_task(tts_consumer())
        
        sentence_buffer = ""
        first_token = False
        
        async for event in graph.astream_events({"messages": [HumanMessage(content=text)]}, {"configurable": {"thread_id": session_id}}, version="v2"):
            if agent_track.stop_requested: break
            if event["event"] == "on_chat_model_stream" and event["metadata"].get("langgraph_node") == "agent":
                chunk = event["data"]["chunk"]
                if chunk.content and not chunk.tool_call_chunks:
                    if not first_token:
                        telemetry.mark("llm_ttft")
                        telemetry.mark("tts_start") 
                        first_token = True
                        
                    chunk_text = chunk.content
                    sentence_buffer += chunk_text
                    await ui_manager.broadcast({"type": "agent_chunk", "text": chunk_text}, room_id)
                    
                    # Sentence Boundary Detection
                    if any(punct in chunk_text for punct in ['.', '!', '?']):
                        if len(sentence_buffer.strip()) > 2:
                            await tts_queue.put(sentence_buffer.strip())
                            sentence_buffer = ""
                            
            elif event["event"] == "on_tool_start":
                await ui_manager.broadcast({"type": "agent_chunk", "text": f"\n*[Searching Database for {event['name']}]*\n"}, room_id)
                
        # Send remaining text and trigger shutdown
        if sentence_buffer.strip():
            await tts_queue.put(sentence_buffer.strip())
            
        await tts_queue.put(None) 
        await consumer_task # Wait for all pending TTS chunks to finish generating
                
        await ui_manager.broadcast({"type": "agent_end"}, room_id)
        asyncio.create_task(telemetry.flush())

        while not agent_track.queue.empty() and not agent_track.stop_requested:
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        agent_track.is_speaking = False
        agent_track.last_speech_end = time.time()
        if not agent_track.stop_requested:
            await ui_manager.broadcast({"type": "state", "val": "Listening..."}, room_id)

async def incoming_audio_processor(track: MediaStreamTrack, session_id: str, agent_track: AgentAudioTrack, room_id: str):
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    vad_iterator = VADIterator(vad_model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=500, speech_pad_ms=80)
    
    _vad_buf = []
    _pre_speech_buffer = collections.deque(maxlen=int(16000/512 * 0.5))
    _vad_started = False
    vad_chunk_buffer = np.array([], dtype=np.float32) 
    
    current_user_delay = 0.0
    
    while True:
        try: frame = await track.recv()
        except: break
            
        for r_frame in resampler.resample(frame):
            data = r_frame.to_ndarray().flatten().astype(np.float32) / 32768.0
            vad_chunk_buffer = np.concatenate([vad_chunk_buffer, data])
            
            while len(vad_chunk_buffer) >= 512:
                chunk = vad_chunk_buffer[:512]
                vad_chunk_buffer = vad_chunk_buffer[512:]
                
                ev = vad_iterator(torch.from_numpy(chunk))
                if ev:
                    if "start" in ev:
                        if agent_track.is_speaking:
                            agent_track.clear_queue()
                            asyncio.create_task(ui_manager.broadcast({"type": "agent_end"}, room_id))
                            asyncio.create_task(ui_manager.broadcast({"type": "system", "text": "⚡ Barge-in detected! Interrupted."}, room_id))
                        
                        _vad_started = True
                        _vad_buf = list(_pre_speech_buffer)
                        current_user_delay = time.time() - agent_track.last_speech_end
                        
                    if "end" in ev and _vad_started:
                        _vad_buf.append(chunk)
                        full_audio = np.concatenate(_vad_buf)
                        _vad_started = False
                        _vad_buf = []
                        
                        audio_duration = len(full_audio) / 16000.0
                        peak_amplitude = float(np.max(np.abs(full_audio)))
                        
                        telemetry = TurnTelemetry(room_id, current_user_delay, peak_amplitude, audio_duration)
                        asyncio.create_task(process_turn(full_audio, session_id, agent_track, room_id, telemetry))
                        
                if _vad_started: 
                    _vad_buf.append(chunk)
                else: 
                    _pre_speech_buffer.append(chunk)

@app.websocket("/ws/webrtc/{room_id}")
async def webrtc_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    pc = RTCPeerConnection()
    session_id = f"{room_id}_{str(uuid.uuid4())}"
    agent_track = AgentAudioTrack()
    pc.addTrack(agent_track)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.create_task(incoming_audio_processor(track, session_id, agent_track, room_id))

    offer = json.loads(await websocket.receive_text())
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await websocket.send_json({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    while True: await asyncio.sleep(1)

@app.get("/")
async def get():
    with open("index.html", "r") as f: return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)