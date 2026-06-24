# Real-Time WebRTC Voice Agent (Alex) with RAG & Live Telemetry

This repository contains a high-performance, low-latency, voice-to-voice agent named **Alex**. The system utilizes **WebRTC** for real-time bidirectional audio streaming and features **Retrieval-Augmented Generation (RAG)** powered by ChromaDB, an inventory SQLite management database, and a live observability telemetry dashboard.

### Demo - [https://share.descript.com/view/rptA3etMY0T](https://share.descript.com/view/rptA3etMY0T)
### P.S - token key is deleted, won't work.
---

## Architecture Overview

The system consists of three main components:
1. **`rtc_server.py`**: The backend server powered by **FastAPI** and **uvicorn**. It orchestrates:
   - **WebRTC connection handling** via `aiortc` (receiving user audio and playing agent audio).
   - **Voice Activity Detection (VAD)** using the Silero VAD model to detect user speech and handle barge-ins.
   - **Speech-to-Text (STT)** using `faster-whisper` (Whisper-Base model).
   - **Brain Logic** using LangGraph and Mistral AI (`ministral-8b-latest`) with tools to search the RAG knowledge base, query/update inventory, and export reports to Excel.
   - **Text-to-Speech (TTS)** using `Kokoro-82M` to synthesize natural-sounding speech chunks in real time.
2. **`ingest.py`**: A CLI utility to populate the ChromaDB vector database with documents (plain text, markdown, or JSONL) to give the agent custom knowledge.
3. **`index.html`**: A front-end interface built with HTML5, TailwindCSS, and native WebRTC APIs, showing live transcripts and a comprehensive telemetry panel monitoring end-to-end latency, STT/LLM/TTS processing times, and audio amplitude.

---

## Prerequisites & System Setup

### 1. System Dependencies (Linux/Debian/Ubuntu)
Because this project builds WebRTC and audio/video processing wrappers, you must have the required system compilation libraries and **ffmpeg** installed on your system. Run:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavdevice-dev libavfilter-dev libavformat-dev \
                       libavcodec-dev libswresample-dev libswscale-dev libavutil-dev \
                       pkg-config python3-dev build-essential
```

### 2. Python Environment Setup & Dependencies (using `uv`)
We highly recommend using **`uv`** for extremely fast virtual environment creation and package installation.

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

*(Alternatively, if using standard `pip`:)*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## API Key Configuration

The voice agent's language model is powered by Mistral AI. You need a valid `MISTRAL_API_KEY` to run the agent. You can configure this key in one of two ways:

### Option A: Set Environment Variable (Recommended)
Export the variable directly in your terminal before running the server:

```bash
export MISTRAL_API_KEY="your_actual_mistral_api_key_here"
```

### Option B: Local Token File
You can place your Mistral API key in a text file named `mistral_token.txt` in the **parent directory** of the project folder. The server will automatically read the token from that file if the environment variable is not set.

---

## RAG Database Ingestion (`ingest.py`)

To enable the agent to answer questions about custom documents (like the research papers located in the `files/` folder), you must ingest them into the ChromaDB vector database.

Use the `ingest.py` utility to load documents:

### 1. Ingest a Single Markdown File (e.g. Reservoir Modeling Review)
```bash
python3 ingest.py --md files/A_Review_of_Reservoir_Modeling_and_Optimization_Me.md
```

### 2. Ingest a Directory of Markdown Files
```bash
python3 ingest.py --dir_md files/
```

### 3. Ingest a Directory of Text/Markdown Files
```bash
python3 ingest.py --dir files/
```

### 4. Ingest Raw Text Strings Inline
```bash
python3 ingest.py --raw "Alex is a customer support agent." "Our business hours are 9 AM to 5 PM."
```

### 5. Check Vector Database Ingest Stats
```bash
python3 ingest.py --stats
```

### 6. Clear/Reset Vector Database (Destructive)
```bash
python3 ingest.py --clear
```

---

## Running the Agent

### Step 1: Start the Backend Server
Run the FastAPI application:

```bash
python3 rtc_server.py
```

Upon startup, the server will download any required local models (Silero VAD, Faster-Whisper, Kokoro TTS pipeline, and Sentence Transformers for RAG embeddings) and print:
`INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

### Step 2: Open the Web UI
1. Open your web browser and navigate to `http://localhost:8000`.
2. Enter a **Room ID** of your choice (e.g., `room_123`) or keep the random default, and click **Join Workspace**.
3. Click **Start Voice Link**.
4. Grant the browser permission to access your microphone.
5. Once the status shows `Voice Link Active! Listening...`, begin speaking!

### Step 3: Accessing from a Separate/Mobile Device (via ngrok)
In modern mobile and desktop browsers, microphone access (`getUserMedia`) is restricted to secure origins (`localhost` or `https://`). If you attempt to connect directly to your computer's local IP address (e.g., `http://192.168.1.5:8000`) from a mobile phone or another device, the browser will refuse to request microphone permissions.

To run the agent and test it from a mobile device or a separate computer, use an **ngrok** tunnel to create a secure `https` gateway:

1. **Install ngrok** (if not already installed):
   Follow the instructions on the [ngrok website](https://ngrok.com/download) or install it via your system's package manager.

2. **Start the Tunnel**:
   With your local backend server running on port `8000`, open a new terminal window and run:
   ```bash
   ngrok http 8000
   ```

3. **Open the HTTPS URL**:
   Copy the secure forwarding URL (which starts with `https://`) provided by ngrok in the terminal (e.g., `https://xxxx-xx-xx-xx.ngrok-free.app`).
   
4. **Interact**:
   Open this URL on your mobile phone or separate device. The browser will allow you to grant microphone permissions over the secure connection, and WebRTC audio and live telemetry will stream successfully.

---

## Diagnostics & Live Telemetry

The right panel of the browser UI displays **Live Telemetry** for each voice turnaround:
- **True E2E Turnaround**: The duration between when you finished speaking and when the agent's voice output started playing.
- **Real Time Factor (RTF)**: Ratio of processing duration to audio length (lower is better).
- **First Audio Latency (TTFA)**: Delay between the VAD trigger and the first synthesized audio stream.
- **STT/LLM/TTS Latency**: Granular breakdowns of the time taken for speech recognition, language model streaming, and speech synthesis.
- **Barge-In Handling**: If you speak while the agent is responding, the Silero VAD will detect it, trigger an interrupt signal, clear the audio playback queue, and immediately listen to your new query.
