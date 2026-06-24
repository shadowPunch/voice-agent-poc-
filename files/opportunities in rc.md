Perspective https://doi.org/10.1038/s41467-024-45187-1
Emerging opportunities and challenges for
the future of reservoir computing
Received:15April2023 MinYan 1,CanHuang 1 ,PeterBienstman2,PeterTino3,WeiLin 4,5&
JieSun 1
Accepted:16January2024
Publishedonline:06March2024
Reservoircomputingoriginatesintheearly2000s,thecoreideabeingto
Checkforupdates utilizedynamicalsystemsasreservoirs(nonlineargeneralizationsofstandard
bases)toadaptivelylearnspatiotemporalfeaturesandhiddenpatternsin
complextimeseries.Showntohavethepotentialofachievinghigher-precision
predictioninchaoticsystems,thosepioneeringworksledtoagreatamountof
interestandfollow-upsinthecommunityofnonlineardynamicsandcomplex
systems.Tounlockthefullcapabilitiesofreservoircomputingtowardsafast,
lightweight,andsignificantlymoreinterpretablelearningframeworkfor
temporaldynamicalsystems,substantiallymoreresearchisneeded.This
Perspectiveintendstoelucidatetheparallelprogressofmathematicaltheory,
algorithmdesignandexperimentalrealizationsofreservoircomputing,and
identifyemergingopportunitiesaswellasexistingchallengesforlarge-scale
industrialadoptionofreservoircomputing,togetherwithafewideasand
viewpointsonhowsomeofthosechallengesmightberesolvedwithjoint
effortsbyacademicandindustrialresearchersacrossmultipledisciplines.
Atthecoreoftoday’stechnologicalchallengesistheabilitytoprocess computers,oftendeviatingfromthevon-Neumannarchitectureand
informationatmassivelysuperiorspeedandaccuracy.Despitelarge- drawinginspirationsfrombiologicalandphysicalprinciples10.Within
scalesuccessofdeeplearningapproachesinproducingexcitingnew thebroaderfieldofneuromorphiccomputing,animportantfamilyof
possibilities1–7,suchmethodsgenerallyrelyontrainingbigmodelsof models known as reservoir computing (RC) has progressed sig-
neuralnetworksposingseverelimitationsontheirdeploymentinthe nificantlyoverthepasttwodecades11,12.RCconceptualizeshowabrain-
mostcommonapplications8.Infact,thereisa growing demandfor likesystemoperates,withacorethree-layerarchitecture(seeBox1
developingsmall,lightweightmodelsthatarecapableoffastinference andBox2):Aninput(sensing)layerwhichreceivesinformationand
andalsofastadaptation-inspiredbythefactthatbiologicalsystems performssomepre-processing,amiddle(processing)layertypically
such as human brains are able to accomplish highly accurate and defined by some nonlinear recurrent network dynamics with input
reliableinformationprocessingacrossdifferentscenarioswhilecost- signalsactingasstimulusandanoutput(control)layerthatrecom-
ingonlyatinyfractionoftheenergythatwouldhavebeenneeded binessignalsfromtheprocessinglayertoproducethefinaloutput.
usingbigneuralnetworks. Reminiscentofmanybiologicalneuronalsystems,thefrontendofan
Asanalternativedirectiontothecurrentdeeplearningparadigm, RCnetwork,includingitsinputandprocessinglayers,isfixedandnon-
research into the so-called neuromorphic computing has been adaptive,whichtransformsinputsignalsbeforereachingtheoutput
attracting significant interest9. Neuromorphic computing generally layer;inthelast,outputpartofanRCthesignalsarecombinedinsome
focusesondevelopingnoveltypesofcomputingsystemsthatoperate optimizedwaytoachievethedesiredtask.Animportantaspectofthe
atafractionoftheenergycomparingagainstcurrenttransistor-based output layer is its simplicity, where typically a weighted sum is
1TheoryLab,CentralResearchInstitute,2012Labs,HuaweiTechnologiesCo.Ltd.,HongKongSAR,China.2PhotonicsResearchGroup,Departmentof
InformationTechnology,GhentUniversity,Gent,Belgium.3SchoolofComputerScience,TheUniversityofBirmingham,BirminghamB152TT,United
Kingdom.4ResearchInstituteofIntelligentComplexSystems,FudanUniversity,Shanghai200433,China.5SchoolofMathematicalSciences,SCMS,SCAM,
andCCSB,FudanUniversity,Shanghai200433,China. e-mail:huangcan321@gmail.com;riosun@gmail.com
NatureCommunications|(2024)15:2056 1
;,:)(0987654321 ;,:)(0987654321

BOX 1
Comparison bettwen deep learning and reservoir computing
Number of Parameters (Billion)
gniniarT
rof
deriuqeR
spolfateP
fo
tnuomA
(a) (b) Memory Taking Up by Parameters (GB)*
Wearable
Phone/PC GPT4
Work Station
HPC
GPT3
T-NLG
MegatronLM
GPT2
RC BERT-Large
Deep Learning GPT
?
(a) The architecture of DL versus RC. For DL, all connections are (b) RC versus DL in terms of the number of parameters and
trainable (denoted by wavy lines). While in RC, only readout computational cost (measured in the amount of petaflops)
weights are trainable, and other connections are fixed once required for training [128].
generated (denoted by straight lines). * Estimation of memory storage of the parameters, assuming
that each parameter is in Float32 format.
Deep learning (DL) and reservoir computing (RC) are both machine learning techniques. They share some
common characteristics. For example, both of them are data-driven frameworks for learning, taking inputs and
transform them (nonlinearly) to match desired outputs. By learning the features from the input data, they are
shown to be universal function approximation, so as to fulfill sophisticated tasks.
However, deep learning and reservoir compuitng are different in some degrees:
1. Architecture design: DL and RC can be distinguished directly from their structures. As shown in Fig. (a), in DL,
all the parameters are fully trainable, namely all connections are continuously updated during the training phase.
While in RC, only the readout weights are trained. Other connections among neurons are fixed once generated
and are not updated any further. This structural difference indicates that RC usually has smaller parameter size
than those of DL.
2. Training procedure: Different architectures determine that DL and RC are trained distinctly. In DL, there have
been many training algorithms and tools developed, such as backpropagation (BP), stochastic gradient descent
(SGD), Newton’s method (NM), and so on. However, in RC, it is simple regression (e.g., linear regression, Lasso
regression and ridge regression) that are usually adopted in training. The small parameter size and simple
training procedure of RC together lead to much less training time and resource consumption.
3. Model complexity & performance: DL and RC have distinct model size, training comlexity and performance. In
Fig. (b), we summrize the parameter size and required training petaflops of DL and RC. As the capacity of deep
learning increases, the parameter size also grows, which is a challenge for practical application. For example,
the memory of smart watch is around typically 2GB, so it can be equipped with GPT (~0.5 GB) or BERT-Large
(~1.3 GB). For large networks such as GPT3 (~652 GB) and GPT4 (~6557 GB), only workstations or high
performance cluster (HPC) can incorporate them. Inversely, since RC has much less parameters, it can be
applied on diverse devices flexibly. Is has been shown that RC can realize image recognition with around 10--55
petaflops [129], indicating wide scope for further explorations. In addition, although parameter size is smaller, RC
is utilized in improving the accuracy in climate modeling [110] and fulfilling weather forecast [111], which had
been realized by deep learning previously.
As one of the most popular machine learning algorithms, DL has been studied widely. Nevertheless, RC seems
to remain at primary stage, no matter in theoretical or algorithm level. It is still an open question where the full
potential of RC is (as indicated by a question mark in Fig. (b)), and how is the training complexity if RC involves
around billions of parameters, for which we draw our hypothesis in dotted lines in Fig (b).
……
……
……
……
……
Perspective https://doi.org/10.1038/s41467-024-45187-1
DL
Fully Trainable
RC
Fixed Trainable
NatureCommunications|(2024)15:2056 2

Perspective https://doi.org/10.1038/s41467-024-45187-1
BOX 2
Schematic representation of the reservoir computing (RC) framework
sufficient, reminding a great deal of how common mechanical and overviewofthecurrentstatusintheoretical,algorithmicandexperi-
electrical systems operate - with a complicated core that operates mentalRCs,toidentifycriticalgapsthatpreventsindustryadoptionof
internallyandacontrollayerthatenablessimpleadaptationaccording RCandtodiscussremedies.
tothespecificapplicationscenario.
Cansuchanarchitecturework?Thisinquirywasattemptedinthe TheoryandalgorithmdesignofRCsystems
early2000sbyJaeger(echostatenetworks(ESNs)11)andMaass(liquid The core idea of RC is to design and use a dynamical system as
statemachines(LSMs),12),achievingsurprisinglyhighlevelofprediction reservoirthatadaptivelygeneratessignalbasisaccordingtotheinput
accuracy in systems that exhibit strong nonlinearity and chaotic dataandcombinestheminsomeoptimalwaytomimicthedynamic
behavior.Thesetwoinitiallydistinctlinesofworkwerelaterreconciled behaviorofadesiredprocess.Underthisangle,wereviewanddiscuss
into a unified, reservoir computing framework by Schrauwen and important results on representing, designing and analyzing RC
Verstraeten13,explicitlydefininganewareaofresearchthattouches systems.
uponnonlineardynamics,complexnetworksandmachinelearning.
ResearchinRCoverthepasttwentyyearshasproducedsignificant MathematicalrepresentationofanRCsystem
resultsinthemathematicaltheory,computationalmethodsaswellas ThemathematicalabstractionofanRCcangenerallybedescribedin
experimental prototypes and realizations, summarized in Fig. 1. the language of dynamical systems, as follows. Consider a coupled
Despitesuccessesinthoserespectivedirections,large-scaleindustry- systemofequations
wide adoption of RC or broadly convincing “killer-applications” (cid:1)
beyondsyntheticandlabexperimentsarestillnotavailable.Thisisnot Δx=Fðx;u;pÞ,
ð1Þ
duetothelackofpotentialapplications.Infact,thankstoitscompact y=Gðx;u;qÞ:
designandfasttraining,RChaslongbeensoughtasanidealsolutionin
many industry-level signal processing and learning tasks including HeretheoperatorΔactingonxbecomesdxforacontinuous-time
dt
nonlinear distortion compensation in optical communications, real- system,x(t+1)−x(t)foradiscrete-timesystem,andacompoundof
time speech recognition, active noise control, among others. For these two operations for a hybrid system. Additionally, u2Rd,
practicalapplications,anintegratedRCapproachismuchneededand x2Rn,andy2Rmaregenerallyreferredtoastheinput,internalstate
canhardlybederivedfromexistingworkthatfocusesoneitherthe and output of the system, respectively, with vector field F, output
algorithmortheexperimentalone.Thisperspectiveoffersaunified function G and parameters p (fixed) and q (learnable) representing
NatureCommunications|(2024)15:2056 3

Perspective https://doi.org/10.1038/s41467-024-45187-1
Fig.1|SelectedresearchmilestonesofRCencompassingsystemandalgorithmdesigns,representingtheory,experimentalrealizationsaswellasapplications.
Foreachcategoryaselectionoftherepresentativepublicationswerehighlighted.
theirfunctionalcouplings.OncesetupbyfixingthevectorfieldFand classical computers, most commonly used RC takes discrete time
theoutputfunctionGandtheparametersp,onecanutilizetheRC steps:
systemtoperformlearningtasks,typicallyintime-seriesdata.Givena (
time series fzðtÞ2Rmg t2N, an optimization problem is usually for- xðt+1Þ=ð1(cid:2)γÞxðtÞ+γfðWxðtÞ+WðinÞuðtÞ+bÞ,
ð3Þ
mulatedtodeterminethebestq: yðtÞ=WðoutÞxðtÞ,
Z
(cid:3) (cid:4)
min kGðxðtÞ;uðtÞ;qÞ(cid:2)zðtÞk2+βRðqÞ dt, ð2Þ whichisaspecialformof(1),butnowwithtimestepsandnetwork
q t parameters more explicitly expressed. In this form, f is usually a
component-wisenonlinearactivationfunction(e.g.,tanh),theinput-
whereR(q)isaregularizationterm. to-internal and internal-to-output mappings are encoded by the
Also,whenz(t)isseenasadrivingsignal,theoptimizationpro- matricesW(in)andW(out),whereastheinternalnetworkisrepresented
blemcanberegardedasadriving-responsesynchronizationproblem bythematrixW.Theadditionalparametersbandγareusedtoensure
finding appropriate parameters q14. Since RC is often simulated on that the dynamics of x is bounded, non-diminishing and (ideally)
NatureCommunications|(2024)15:2056 4

Perspective https://doi.org/10.1038/s41467-024-45187-1
exhibitsrichpatternsthatenablelaterextraction.Givensometraining entire class of problems and ask what might be the best RC archi-
time series data {z(t)} (assumed to be scalar for notational conve- tecture - including its input and internal coupling dynamics and
nience),oncetheRCsystemissetupbyfixingthechoiceoff,γ,b,W(in) trainingobjective25,26,forinstanceusingBayesianoptimization27.Fur-
andW,theoutputweightmatrixW(out)canbeobtainedbyattempting thermore, nonlinearfunctions beyond the component-wise f=tanh
tominimizealossfunction.Acommonlyusedlossfunctionis areoftenencounteredinexperimentalsettingsandanactivelineof
researchistoexplorenewtypesofnonlineardynamicssuchaselectro-
(cid:3) (cid:4)
WðoutÞ>=argmin kXw(cid:2)zk2+βkwk2 , ð4Þ optic phase-delay dynamics28–30, optical scattering31,32, dynamic
w memristors33–36, enlarged memory capacity in chaotic dynamics37,
where X=ðxð1Þ>,xð2Þ>,...,xðTÞ>Þ > , z=(z(1),z(2),…,z(T))⊤ and solitons38andquantumstates39,40.
β∈[0,1]isaprescribedparameter.Thisproblemisinaspecialformof
Tikhonov r(cid:5)egularizatio(cid:6)n and yields an explicit solu- MathematicaltheorybehindRC
(cid:2)1
tionWðoutÞ>= X>X+β2I X>z. Thefundamentalquestionsofexactlywhy,whenandhowRClearnsa
general dynamical process are important mathematical questions
CommonRCdesigns whoseanswersare expected to provide guidelines for the practical
DesigningisacrucialstepforacquiringapowerfulRCnetwork.There designandimplementationofRCsystems.Theselinesofquerieshave
arestillnocompleteinstructionsonhowtodesignoptimalRCnet- ledtoanumberofimportantanalyticalresultswhichweclassifyinto
worksbasedonvariousnecessities.WiththeunifiedformsEqs.(1)and fourcategories.
(2) in mind, a standard RC system as initially proposed contains Thefirstcategoryofworkfocusesonthe echostateproperty
everythingrandomandfixedincludingtheinputandinternalmatrices (ESP):Equivalenttostatecontracting, stateforgetting, andinput
W(in) and W, leaving the choice of parameters γ and β according to forgetting-referstoRCnetworkswhoseasymptoticstatesx(t→∞)
someheuristicrules.Basedonthisdefaultsetting,weshowhowdif- dependsonlyontheinputsequenceandnotontheinitialnetwork
ferentRCdesignscangenerallybeinterpretedasoptimizinginone states.Thispropertyleadstoacontinuitypropertyofthesystem
and/or multiple parts along the following directions. Firstly, in RC knownasthefadingmemorypropertywherecurrentstateofthe
coupling parameter search, with the goal of selecting a good and system mostlydepends on near-termhistoryand notlongpast11.
potentiallyoptimalcouplingparameterγtomaintaintheRCdynamics Ref. 11 considers RC network with sigmoid nonlinearity and unit
boundedandproducesrichpatternthatallowfortheinternalstatesto outputfunctionandshowedthatifthelargestsingularvalueofthe
form a signal bases that can later be combined to approximate the weightmatrixWislessthanonethenthesystemhasESP,andifthe
desiredseries{z(t)}.Empiricalstudieshaveshownthatγchosensothat spectralradiusofWislargerthanonethenthesystemisasymp-
thesystemisaroundtheedgeofchaos15typicallyproducesthebest totically unstable and thus cannot has ESP. Tighter bounds were
outcome,whichissupportedbyanecessarybutnotsufficientcondi- subsequentlyderivedin41.Inparticular,thespectralradiuscondi-
tion-imposedonthelargestsingularvalueoftheeffectivestability tionprovidesapracticalwayofrulingoutbadRCsandcanbeseen
matrix Wγ=(1−γ)+γW. Then, in RC output training, whose design anecessaryconditionforRCtoproperlyfunction.
commonly amounts to two aspects. One is to determine the right Thesecondcategoryisaboutmemorycapacity.Definedbythe
optimizationobjective,forinstancetheoneinEq.(4)withcommon summation of delay linear correlations of the input sequence and
generalizationsincludetochangethenormsusedintheobjectivein outputstates,wasshowntonotexceedNforunderiidinputstream42,
particularthe term ∥w∥to enforce sparsity or toimpose additional canbeapproachedwitharbitraryprecisionusingsimplelinearcyclic
prior information by changing β∥w∥ into ∥Lw∥ with some matrix L reservoirs16,andcanbeimprovedusingthetimedelaysinthereservoir
encodingthepriorinformation.Ontheotherhand,(uponchoiceof neurons43.
theobjective)tofurtherdeterminetheparameter,e.g.,βasinEq.(4). Universal approximation theorems can be regarded as a single
Althoughthereisnogeneraltheoreticallyguaranteedoptimalchoice, category.PriortotheresearchofRC,universalrepresentationtheo-
severalcommonmethodscanbeutilized,e.g.,cross-validationtech- remsbyBoydandChua showedthatanytime-invariantcontinuous
niquesthathadbeenwell-developedintheliteratureofcomputational nonlinearoperatorcanbeapproximatedeitherbyaVolterraseriesor
inverse problems. RC network design is crucial to determine the alternatively by a lineardynamical system with nonlinear readout44.
dynamiccharacteristics.Withthegoalofdeterminingagoodinternal RC’s representation power has attracted significant recent interest:
coupling network W. This has received much attention and has ESNs are shown to be universally approximating for discrete-time
attractedmanynovelproposals,whichincludestructuredgraphswith fadingmemoryprocessesthatareuniformlybounded45andfurther
random as well as non-random weights16,17, and networks that are thattheapproximatingfamilycanbeassociatedwithnetworkswith
layered and deep or hierarchically coupled18–20. Furthermore, some- ESPandfadingmemory46.Fordiscrete-timestochasticinputs,linear
timesthosedesignsarethemselvescoupledwiththewaytheinputand reservoirsystemswitheitherpolynomialorneuralnetworkreadout
output parts of the system are used, for example in solving partial mapsareuniversalandsoareESNswithlinearoutputsunderfurther
differential equations (PDEs)21,22 or representing the dynamics of exponentialmomentconstraintsimposedontheinputprocess47.For
multivariate time series23. Finally, as for RC input design, although structurallystablesystems,theycanbeapproximated(upontopolo-
receivedrelativelylittleattentionuntilrecently,itturnsoutthatthe gicalconjugacy)byasufficientlylargeESN48.Inparticular,ESNswhose
inputpartofanRCcanplayveryimportantrolesinthesystem’sper- outputstatesaretrainedwithTikhonovregularizationareshownto
formance. Hereinput design is generally interpreted toinclude not approximate ergodic dynamical systems49. Also rigorously, the
onlythedesignoftheinputcouplingmatrixW(in)butalsopotentially dynamicsofRCisvalidatedasahigher-dimensionalembeddingofthe
some(non)lineartransformationontheinputu(t)and/ortargetvari- input nonlinear dynamics43. In addition, explicit error bounds are
ablez(t)priortosettinguptherestoftheRCsystem.Theso-called derivedforESNsandgeneralRCswithESPandfadingmemoryprop-
next-generation RC (NG-RC) is one such example24, showing great erties under input sequences with given dependency structures50.
potentialofinputdesigninimprovingthedataefficiency(lessdata Finally, according to conventional and generalized embedding the-
requiredtotrain)ofanRC. ories, the RCs with time delays are established with significantly-
InadditiontotheseparatedesignsoftheindividualpartsofanRC, reducednetworksizes,andsometimescanachievedynamicsrecon-
thenovelconceptofneuralarchitecturesearch(NAS)hasmotivated structioneveninthereservoirwithasingleneuron43.
the research of hyperparmeter optimization25 and Automated RC Thelastcategoryincludesresearchaboutlinearversusnonlinear
design to (optimally) design an RC system for not just one, but an transformationsandnext-generationRC.Focusingonlinearreservoirs
NatureCommunications|(2024)15:2056 5

Perspective https://doi.org/10.1038/s41467-024-45187-1
BOX 3
Schematic diagram of physical reservoir computing
(possiblyuponpre-transformationsoftheinputstates),recentwork discoveriesweremadeestablishinguniversalapproximationtheorems
showedthattheoutputstatesofanRCcanbeexpressedintermsofa ofRC-thoseresults,althoughnotyetdirectlyusefulforconstructing
controllability matrix together with the network encoded inputs17. optimalRCs,mayneverthelessboostconfidenceandstimulateexpli-
Moreover, a simplified class of RCs are shown to be equivalent to citlyideasofdesigningandevenoptimizingRCsforlearning.Inpar-
generalvectorautoregressive(VAR)processes51-withpossiblenon- ticular,despitehavingrandomlyassignedweightsthatarenottrained,
linear basis expansions it forms theoretical foundations for the RCmodelsareneverthelessshowntopossessstrongrepresentation
recentlycoinedconceptofnext-generationRC24. powerwithrigoroustheoreticalguarantees.
ResearchofhowtodesignRCarchitectures,howtotrainthem
and why they work have, over the past two decades following the PhysicaldesignofRCsystems:fromintegrated
pioneeringworksofJaegerandMaass,ledtomuchevolvedviewofthe circuitstosiliconphotonics
capabilitiesaswellaslimitationsoftheRCframeworkforlearning.On To archive a controllable nonlinear high-dimensional system with
theonehand,simulationandnumericalresearchhasproducedmany short-term memory, some specific physical systems with nonlinear
newnetworkarchitecturesimprovingtheperformanceofRCbeyond dynamic characteristics can be used to implement reservoirs
purelyrandomconnections;futureworkscaneitheradoptaone-fits-all (seeBox3),wherenetworkconnectionsaredeterminedbythephy-
approachtoinvestigateverylargerandomRCsorperhapsmorelikely sicalinteractions.Asthedevelopmentofintegrationtechnologyfor
to follow the concept of domain-specific architecture (DSA)52 to electricalandopticalcomponent,thecomputationalefficiencycanbe
explorestructuredclassesofRCsthatachieveoptimalperformance greatlyimprovedcomparedtotraditionalBooleanlogicmethods.The
forparticulartypesofapplications,withBayesianoptimization26,27and implementation of physical reservoir is similar to the software
NASaspowerfultoolsofinvestigation53.Ontheotherhand,foralong approach,butslightlydifferent.
timeonlyfewtheoreticalguidelinesbasedonESPwereavailablefor Inrecentyears,therehasbeenextensiveresearchondesigning
practicaldesignofRCs;morerecentlyseveralimportanttheoretical and realizing RC using physical systems. A detailed review can be
NatureCommunications|(2024)15:2056 6

Perspective https://doi.org/10.1038/s41467-024-45187-1
331,231,47,37 foundinreference54.Physicalreservoirscanberoughlydividedinto
87,63,53,33
|     | three | types | based | on their | topological |     | structure: | discrete | physical |     |
| --- | ----- | ----- | ----- | -------- | ----------- | --- | ---------- | -------- | -------- | --- |
131,031,17 431,97,56 531,86,85 721,23,13
|     | nodes | reservoir, | single-node |     | reservoirs |     | with delayed |     | feedback | and |
| --- | ----- | ---------- | ----------- | --- | ---------- | --- | ------------ | --- | -------- | --- |
68,66
.sfeR continuousmediumtypereservoirs.Discretephysicalnodesreservoir
|     | is            | composed | of             | interacting |                | nonlinear | components, |          | such | as  |
| --- | ------------- | -------- | -------------- | ----------- | -------------- | --------- | ----------- | -------- | ---- | --- |
|     | memristors35, |          | spintronics55, |             | oscillators56, |           |             | nodes32, |      |     |
.sedonraenilnonehtfotimilemitesnopserehthcaernacdeepsgnitarepoeht,yllaedI.srotcafrehtodnaemitesnopserraenilnonedon,noisrevnocA/D-D/A,gnissecorperpatadybdecneuflnisimetsysehtfodeepsgnitarepoehT.1etoN optical etc. The
|     | nodes | form | a coupling | network |     | through | real physical |     | connections. |     |
| --- | ----- | ---- | ---------- | ------- | --- | ------- | ------------- | --- | ------------ | --- |
cirtcelE-cirtcelE cirtcelE-cirtcelE cirtcelE-cirtcelE cirtcelE-cirtcelE -lacitpO-cirtcelE -lacitpO-cirtcelE -lacitpO-cirtcelE
|     | They | can be | simply | enlarged | by  | increasing | the | number | of network |     |
| --- | ---- | ------ | ------ | -------- | --- | ---------- | --- | ------ | ---------- | --- |
elementstoobtainhigherdimensions.Single-nodereservoiriscom-
sepyTO/I
-cirtcelE -cirtcelE -cirtcelE posed of a singlenonlinearnode and a time delayloop, which can
|                  | transform |      | the input | signal         | into | a virtual        | high-dimensional |           |              | space |
| ---------------- | --------- | ---- | --------- | -------------- | ---- | ---------------- | ---------------- | --------- | ------------ | ----- |
|                  | through   | time | division  | multiplexing   |      | using            | single           | nonlinear | physical     |       |
| 2ytilibA-margorP |           |      |           | circuits57,    |      | lasers58,        |                  |           |              |       |
|                  | nodes,    | such | as analog |                |      |                  | etc. This        | type      | of reservoir |       |
|                  | avoids    | the  | problem   | of large-scale |      | interconnection, |                  | making    | it           | more  |
hardwarefriendly.However,designingandimplementingappropriate
muideM muideM
|     | delayed | feedback |     | loops | is not | a simple | task. | Continuous-medium |     |     |
| --- | ------- | -------- | --- | ----- | ------ | -------- | ----- | ----------------- | --- | --- |
hgiH woL hgiH hgiH hgiH
reservoirmainlyutilizesthephysicalphenomenaofvariouswavesina
|     | continuous |     | medium, | such | as fluid | and | elastic | media. | This type | of  |
| --- | ---------- | --- | ------- | ---- | -------- | --- | ------- | ------ | --------- | --- |
ycneicfifEygrenE
physicalsystemcanutilizethephysicalpropertiesofwaves,suchas
|     | interference, |     | resonance, |     | and synchronization, |     | to  | achieve | extremely |     |
| --- | ------------- | --- | ---------- | --- | -------------------- | --- | --- | ------- | --------- | --- |
efficientphysicalRC59.Intermsofspecificphysicalschemes,thereare
Wm~Wμ Wm~Wμ
Wm~ Wm~ Wm~ also physical reservoirs implemented by mechanical60, biological61,
Wμ~ Wμ~
quantumsystems39andsuperconductors62.Inthisarticle,wemainly
|     | focus | on comparing |     | various | physical |     | implementation |     | solutions | in  |
| --- | ----- | ------------ | --- | ------- | -------- | --- | -------------- | --- | --------- | --- |
1deepSgnitarepO
|     | terms | of integration, |     | power | consumption, |     | processing |     | speed, | and |
| --- | ----- | --------------- | --- | ----- | ------------ | --- | ---------- | --- | ------ | --- |
programmability,asshowninTable1.Typicalhigh-performancephy-
sicalreservoirsincludetraditionalelectronicschemesrepresentedby
BooleanlogiccircuitssuchasFPGA63andASICs64;Non-VonNeumann
zHM~ zHM~ zHM~ zHM~ zHT~ zHG~ zHG~
|     | electrical    |     | reservoir | scheme |          | represented | by          | memristor33 |     | and     |
| --- | ------------- | --- | --------- | ------ | -------- | ----------- | ----------- | ----------- | --- | ------- |
|     | spintronics65 |     | devices;  | And    | photonic | schemes     | represented |             | by  | silicon |
sedoNforebmuN
photonics66,fiberoptics29,67,68andfree-spaceoptics69.
Inprinciple,existingmorphologicalcircuits,suchasFPGAsand
ASICs,canbeimplementedasanelectronicreservoir.Withitsbit-level
301~01 301~01 201~01 201~01 fine-grained customized structure, parallel computing ability, and
201~ 201~ 401~
efficient
|     |     | energy | consumption, |     |     | FPGAs exhibit | unique |     | advantages | in  |
| --- | --- | ------ | ------------ | --- | --- | ------------- | ------ | --- | ---------- | --- |
sdohtemnoitatnemelpmiCRlacisyhplacipytneewtebnosirapmoC|1elbaT
|     | deep | learning | applications. |     | Using | FPGAs | for reservoir |     | computing | is  |
| --- | ---- | -------- | ------------- | --- | ----- | ----- | ------------- | --- | --------- | --- |
resal;noitprosbaelbarutas;AOS;tceffecirtceleotohP
setaGnaelooBdradnatsnodesabstiucricraenilnoN alsoadvantageous,assparseconnectionsinthereservoirmodelallow
|     | for         | simple  | routing  | techniques  | that | match        | FPGA           | requirements. |              | Cur-   |
| --- | ----------- | ------- | -------- | ----------- | ---- | ------------ | -------------- | ------------- | ------------ | ------ |
|     | rently,     | several | FPGA     | methods     | have | been         | proposed70–72. |               | In addition, |        |
|     | considering |         | the high | programming |      | requirements |                | of FPGAs,     |              | people |
haveproposedtheimplementationofRCalgorithmusingApplication
| scimanydlacirtcele-nipsraenilnoN | Specific |            |     |          | (ASICs)73, |       |     |      |         |      |
| -------------------------------- | -------- | ---------- | --- | -------- | ---------- | ----- | --- | ---- | ------- | ---- |
|                                  |          | Integrated |     | Circuits |            | which | can | help | improve | chip |
performanceandpowerconsumptionratio.ThedisadvantageofASIC-
basedRCisthatcircuitdesigncustomizationleadstorelativelylong
.defiidomdnadeniartebnacriovreserafiybdenimretedsiytilibammargorP.2etoN
developmentcycles,inabilitytoscale,andhighcosts.Butresearchin
| ecnatsisercitats-noN | thisareaisalsoactivelyadvancing74,75. |     |     |     |     |     |     |     |     |     |
| -------------------- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ytiraenilnonedoN BesidestheelectricreservoirthatisbasedonBooleanlogicand
|     | von-Neumann |     | architecture, |     | people | have | been | pursuing | higher | effi- |
| --- | ----------- | --- | ------------- | --- | ------ | ---- | ---- | -------- | ------ | ----- |
scimanyd ciency and lower energy consumption methods. For the reservoir
model,thenonlinearanalogelectroniccircuitcanbeusedtodirectly
buildthereservoirmodel,suchastheMackey-Glasscircuit76.Basedon
|     | nonlinearelectronic |     |     | circuits, | asingle | electricnode,such |     |     | asa | mem- |
| --- | ------------------- | --- | --- | --------- | ------- | ----------------- | --- | --- | --- | ---- |
metsyslacisyhP scinotohpnociliS scitpoecapseerF ristor,oraspintronicdevice,withdelaylinesthatcanbeconstructed
|     | and | combined | with | other | digital | hardware | components |     | for | pre- |
| --- | --- | -------- | ---- | ----- | ------- | -------- | ---------- | --- | --- | ---- |
scitporebiF
rotsirmeM scinortnipS processingandpost-processing77.Thememristorhasthedimensionof
resistance,butitsresistancevalueisdeterminedbythechargeflowing
sAGPF sCISA
throughit.Itfunctionsasamemory,andcangeneraterichreservoir
statesunderanappropriatetimedivisionmultiplexingmechanism35.In
|     | addition, | it  | is also | possible | to  | construct | 2D/3D | memristor | crossbar |     |
| --- | --------- | --- | ------- | -------- | --- | --------- | ----- | --------- | -------- | --- |
semehcsriovreseR
stiucricdetargetnI arrays and encode matrix elements into the embedded memristor
smetsyscinotohP
raenilnonlevoN conductance78.Thisprogrammingcanbeaccomplishedusingvoltage
pulseswithminimalenergyrequired.Ontheotherhand,micro/nano
stiucric spin electronic devices constructed using electron spin degrees of
freedomcanexhibitthephysicalpropertiesoftinymagnetsandcanbe
usedtosimulatesynapticbehaviorinbiologicalnerves65.Atpresent,
7
NatureCommunications|(2024)15:2056

Perspective https://doi.org/10.1038/s41467-024-45187-1
peoplehaveproposedseveralreservoirschemesbasedonthephysical digits16,28,29,34,35,65,68,82–84,imagelabels33,35,85–87,bit-symbols16,29,68,88–93and
phenomenarelatedtospintronics79. soon.Theeffectivenessoftraditionalneuralnetworksinclassification
On the other hand, development of photonic technology has taskshasbeenverifiedinlotsofwork.However,dealingwithtemporal
brought hope for ultra-high speed and low energy consumption inputsignalisstillachallenge.Comparedwithtraditionalneuralnet-
hardware systems, especially for neural network training80. Optical works,RCcanmaptemporalsignalswithmultipletimescalestohigh
systemshavesignificantadvantagesovertraditionalmicroelectronic dimension, encoding these signals with its various internal states.
technologiesintermsofhighbandwidth,lowlatency,andlowenergy Furthermore,RCnetworkhasmuchlessparametersthusrequiringless
consumption.Reservoirnetworksbasedonopticalsystemshavealso trainingresources.Therefore,RCcanbeagoodcandidatetobeuti-
made significant progress81, such as multi-scattering nodes in free lizedintemporalsignalclassificationtasks.Thesignalsareinvarious
space32,singlenonlinearnodeswithfiberloop58,andintegratedon- types (audio, image or temporal waves), and usually require some
chipreservoirs66.Thefree-spacereservoirisgenerallyachievedusing preprocessing before injecting to RC network. For example, in the
spatial optics and scattering media, such as diffractive optical ele- spoken-digit recognition task, the raw signal is first transformed to
ments (DOE), to achieve coupling between spatial optical nodes. frequencydomainintermsofmultiplefrequencychannelsviaLyon’s
Interconnectionbetweenneuronsinthereservoirarerealizedthrough passiveearmodel,asshowninFig.2a.Thenthe2-Dsignalscanbe
complexscatteringprocesses32.Singlenonlinearopticalnodes,suchas directlymappedtotheRCnetworkasinputu(t)viainputmask,orcan
semiconductor optical amplifiers (SOAs), saturable absorbers be transformed to 1-D input sequence u(t) by connecting each row
(SESAM),aswellassemiconductorlaserscanformopticalreservoirs successively.Thetargetsareavectorofsizetencorrespondingtodigit
withspecialfiberloopdesigns81.Integratedon-chipopticalreservoir numberfrom0to9.Thestate-of-the-artofRCcurrentlycanreacha
are often archived by interaction between nonlinear micro/nano worderrorrate (WER) of0.4%from memristorchip RC35,and0.2%
optical devices, such as micro-rings81. Unlike the fiber delay loop fromelectronicRC28.
architecture,utilizingmultipleonchipnonlinearopticalnodesmakes For time series prediction, RC assumes the role of regression,
itmoreconvenienttotakeadvantageofopticalparallelcomputing. takinginputasasegmentoftimeseriesuptoacertaintimeanddraws
Comparativelyspeaking,reservoirschemesbasedonFPGAsand predictions for the next (few) time steps. Examples are abundant,
ASICs can greatly improve the computing speed and power con- including prediction of chaotic dynamics such as Mackey-Glass
sumptioncomparedwiththegeneralCPUelectronicarchitecture,due equations11,31,34,51,94, Lorenz system22,26,49,51,94–96, Santa Fe Chaotic time
toitsnon-VonNeumann/in-memorynatureofthecomputing.Besides, series16,86,89,94, Ikeda system94, auto-regressive moving average
there is no need for photoelectric conversion at either the input (NARMA)sequence16,28,29,93,97,98,Hénonmap16,35,94,98,radarsignal68,lan-
oroutputends,makingitconvenientindatascalingandprocessing. guage sentence36, stocks data61, sea surface temperatures (SST)99,
However,thecomputingefficiencyisclosetothetheoreticallimit.For trafficbreakdown100–102,toolweardetection96andwindpower103.Given
electricalnon-VonNeumannarchitectures,suchasmemristors,more atrainingtimeseriesfzðtÞg t2Zandprescribedpredictionhorizonτ,the
efficientcomputationcanberealizedtheoretically,butduetotheir input sequence of RC can be defined as u(t)=z(t) while the target
analognature,itisusuallydifficulttorealizeidealnonlinearmappings outputasy(t)=z(t+τ).(Forone-steppredictionweuseτ=1.)Oncethe
andhigh-precisionmatrixcalculations,andtheintegrationandstabi- parametersofRCislearned,itcanbeusedasapredictivemodel,taking
lity of such devices also need to be improved. As for spintronics atemporalinputandpredictsitsnextsteps.Inparticular,RCtrained
reservoirs,sofar,moststudieshaveonlyexplorednanomagneticRCin withone-steppredictioncanneverthelessbeusedtomakemulti-step
simulations,anditalsofacesthescalabilityproblemsimilartomem- predictions, in the following way. Suppose that a finite-length time
ristor. For optical reservoir schemes, the low delay and low energy series{u(t)} t=1,…,T isprovided,wefeeditintoRCtocomputeastate
consumption characteristics of optical devices are generally only y(t+1)asaone-stepprediction.Wethenappendthisstatetotheendof
reflectedinthereservoirlayer.Currently,mostschemesrequirepho- theinputeffectivelydefiningu(t+1)=y(t+1)andthroughRCtocom-
toelectricconversionindatapreprocessingandpost-processing,and puteanextstatey(t+2),andsoonandsoforthtoobtainaseriesof
the response time of the system is essentially limited by photo- nextstepsy(t+1,t+2,…,t+h).Aschematicexampleofnonlineartime
detectorsandthetimedelayofelectroniccontrolcircuits.Atthesame seriespredictiontaskisshowninFig.2b.Inordertorealizelong-term
time,opticalprocessingerrorsandthepowerconsumptionofexternal prediction, there is another training scheme in which the target
auxiliarydevicesalsoposestrictlimitationsonthescaleofthesystem. sequenceisinsertedperiodically.Inparticular,theinputtotheRCnow
Soitseemsthereiscurrentlynosolutionthatcanbesaidtobethebest. comesfromitsfeedbackortargetsequencealternately,asshownin
Intheshortterm,electronicsolutionssuchasFPGAsdonotrequire Fig. 2b (method 2). Compared with the previous case which can be
photoelectricconversionintheinputandoutputprocesses,andare regardedasanofflinetrainingscheme,hereRCcanacquiretargetdata
measurement friendly, thus having advantages in hardware imple- periodically,thenretrainingandupdatingtheoutputweightsregularly.
mentation.However,consideringissuessuchaspowerconsumption Thisisanonlinetrainingscheme.SinceRChasaccesstotargetdata
andlatency,specializedphotonicreservoirswillhavemoreadvantages duringitsevolution,itcanadjusttheoutputweightstopreventthe
in the future. Perhaps utilizing their respective advantages compre- predictiveoutputdatafromdiverging.Therefore,theonlinetraining
hensivelyandadoptingaheterogeneousintegrationsolutionisafea- typically yields longer prediction period and better prediction
siblepath. performances.
RCcanplayimportantrolesinthecontrolofnonlineardynamical
ApplicationbenchmarksofRC systems104–109. In particular, in the model predictive control (MPC)
ApplicationsofRCarequitediverseandcanbemainlydividedinto framework,controlactionsarederivedbasedonapredictivemodelof
severalcategories:signalclassification(e.g.,spokendigitrecognition), thesystemdynamics.Thepredictivemodelistypicallylineardueto
timeseriesprediction(e.g.,chaospredictionsuchasintheMackey- simplicity and low computational cost. RC as an alternative can
Glassdynamics),controlofsystemdynamics(e.g.,learningtocontrol potentiallyimprovesuponthelinearpredictionwithoutintroducing
robots in real-time) and PDE computations (e.g., fast simulation of toomuchadditionalcomputationaloverhead,asshowninFig.2c.Asa
Kuramoto-Sivashinsky equations), which we discuss below concreteexample,inthecontrollingrobotarmmovementtask104,the
respectively. mechanicalarmgivesinputdatasuchasjointarmangles,destination
Insignalclassificationtasks,theinputofRCareusuallybroadly- positioncoordinatesandjointarmtorquescalculatedfromLagrangian
interpreted(physical)signalssuchasaudio,imageortemporalwaves. equation. RC is trained with the targets which are successive joint
Thetargetoutputarethecorrespondinglabelswhichcanbespoken torquesneededtograduallymovetothedestination.Inthetesting
NatureCommunications|(2024)15:2056 8

Perspective https://doi.org/10.1038/s41467-024-45187-1
Fig.2|ExampleapplicationsofRC.FlowdiagramsshowinghowRCisappliedin andtestingarealternatelypresented.cRCactsasthepredictionoptimizerinthe
differenttypesofapplications,herereferringtoassignalclassification,nonlinear generalmodelpredictivecontrol(MPC)104–109framework.Top:TheMPCdiagram.
timeseriesprediction,dynamicalcontrolandPDEcomputing,respectively.aRC Bottom:HowRCworksintheMPCsystem.dRCforPDEcomputation21,22,32withthe
forspoken-digitrecognition16,28,29,34,35,65,68,82–84,whenthetargetsareavectorofdigit Kuramoto-Sivashinsky(KS)equationsasanexample.Thehiddenlayerconsistsof
numberscorrespondingto0–9.bRCfortimeseriespredictionwithMackey-Glass parallelmultiplereservoirs,andeachofthemdealwithpartoftheinputdata,while
equations11,31,34,51,94asanexample.Inmethod1withoff-linetraining,thetraining anonlineartransformationistypicallyinsertedbeforetrainingtheparametersof
sequencestartswiththefirstpoint(blackpoint),whilethetargetsequencestarts thereadoutlayer.
withthesecondone(orangepoint).Inmethod2withon-lineretraining,thetraining
phase,aslongasthedestinationisgiven,therobotarmcanevolveby improveresults.AsummaryoftrendsinRCperformancesintypical
itselftoapproximatetothetargetpoint.Additionally,RCjointlywith application scenarios is shown in Fig. 3. For example, in spoken
adaptivefeedbackcontroltechniquecanbeusedtotracktheunknown digit recognition, WER is reaching near-perfect levels (0.014%)58.
andunstableperiodicorbitsandstabilizethemevenwhenthechaotic Similarly, handwritten digit recognition boasts an accuracy of
timeseriesareonlyavailable14. around 97.6%35. While RC currently has limitations in action
RC can be applied for scientific computation such as in the recognition and requires preprocessing, there is potential for
numerical solution of PDEs21,22,32. For these tasks, RC is typically future development in expanding recognition abilities and redu-
used to evolve the states of the system toward the temporal cing preprocessing needs. In time series prediction, RCexcels in
direction with a flow diagram shown in Fig. 2d. To decrease the chaoticsequencessuchasMackey–Glass,Lorenz,andSantaFe,but
difficultyforasinglereservoirtoprocessallinputsandimprovethe real-worlddatasuchasweather110,111,stocks,andwindpowershow
trainingefficiency,parallelreservoirarchitecturewasproposed21,22 less impressive performance. RC is primarily used in dynamic
whichallowsmultiplesmallreservoirstodealwithdifferentparts controlforMPCsystems,butassystemcomplexityincreases,real-
ofinputdata.Theinputvectorissplitintomultiplesmallgroups time control with greater accuracy and efficiency is necessary.
with each group includes some extra adjacent points serving as Lastly,RChasbeenshowntocomputePDEseffectively,butprac-
extrainformationprovidedtothecorrespondingsmallreservoirs. ticalapplicationsofthisabilityhaveyettobefullyrealized.Despite
Thetargetis thenexttime step vectorofthePDE.Accompanied attempts and preliminary successes in applying RC to problems
with anonlinearreadoutfunction,theRCnetworkcan learn and endowedwithreal-worlddatasets,nearlynoneofthoseattempts
evolveKuramoto-Sivashinsky(KS)equationrelativelyaccurateup haveledtoanindustry-leveladoptionandapplication.Animpor-
toatimelengthofaround5Lyapunovtimes21. tantreasonisthattheperformanceofRConcommontaskssuchas
Overall, RC has demonstrated strong performance across a imageclassification,audiosignalprocessing havenotreachedor
range of benchmarks and tasks, with ongoing efforts to further showntohavethepotentialtoapproachtheSOTAmetricsoffered
NatureCommunications|(2024)15:2056 9

Perspective https://doi.org/10.1038/s41467-024-45187-1
Fig.3|TrendsinRCperformanceintypicalapplicationscenarios.Fourkindsof LorenzsystemsaswellasandSantaFechaotictimeseries;ccontroltasksanddPDE
representativescenariosare:asignalclassificationtaskssuchasspoken-digit computation.Thick,up-pointingarrowsinthepanelsdenoteerrorvaluesthatare
recognition,nonlinearchannelequationandopticalchannelequalization;btime notdirectlycomparablewithotherworks.
seriespredictionsuchaspredictingthedynamicsofMackey-Glassequations,
bydeep-learning basedmethods.Given thattheoreticallyRChas thatarebothdynamicandlightweight,yetwidelydeployableat
universalapproximationcapacitiesjustasgeneralneuralnetworks, lowcost. According toestimates,bythe years2030–2035,both
inprinciplenothingseemstobeholdingbackRCmodelstopush wireless and optical communication will usher in the sixth gen-
thefrontiersofmostchallengingAItasks,andthisshouldbeamain eration (6G/F6G), providing connections for tens of billions of
goaloftheentireRCcommunity. devicesandmulti-billionusers112,113.Itisalsoexpectedthatglobal
datacenterswillhaveathroughputoftrillionsofGBandrequire
Opportunitiesandtechnicalchallengesforfuture over 200terawatthours ofpower consumption114.Furthermore,
developmentofRC tensandhundredsofmillionsofrobotsaresettoenterourdaily
WeexpectthatresearchinRCcanplayimportantrolesinseveral livestoimprovelaborefficiencyatalowcost115.Virtualrealityand
important application domains, which we discuss as follows. As Metaverse rely heavily on real-time simulation of the physical
technology continues to rapidly advance, there is an increasing world116. These major applications require a large number of
demand to develop intelligent information processing systems capabilities, including accurate recognition of dynamic
NatureCommunications|(2024)15:2056 10

Perspective https://doi.org/10.1038/s41467-024-45187-1
Fig.4|ApplicationdomainsinwhichRCpotentiallycanplayimportantroles. Things(IoT)61,144,145,GreenDataCenter120,146,147,IntelligentRobots148–150andAIfor
Eachdomaincorrespondstothreespecificexampleapplicationscenarios.Six Science151–157andDigitalTwins99,103,158–161.
domainsare6G136–140,NextGeneration(NG)OpticalNetworks92,93,141–143,Internetof
uncertainty information, fast prediction and computation, and interfaces,andlow-complexitytrainingandcomputingnature,RCis
dynamiccontrol,allofwhichcanbeprovidedbyRCsystems,as expectedtobecomeakeytechnologybaseforedge-sideinformation
showninFig.4.Asaresult,weexpectthatRCresearchwillplaya processing.
criticalroleinseveralimportantapplicationdomains,aswewill
discussbelow. Next-generationopticalnetworks
Opportunities.Opticalfibercommunicationisoftenregardedasone
6G ofthemostsignificantscientificadvancementsofthe20thcentury,
Opportunities.Itispredictedthatby2030,wirelesscommunication asnotedbyCharlesKuenKao,theNobelPrizewinnerinPhysics117.
willadvancetoitssixthgeneration,commonlyreferredtoas6G.The The optical network derived from optical fiber technology has
maingoalfor6Gistoenhanceimportantindicatorssuchastrans- become a fundamental infrastructure that supports the modern
missionspeed,coveragedensity,timedelay,andreliabilityby10to informationsociety,processingmorethan95%ofnetworktraffic.
100timescomparedto5G.Thiswouldprovideanever-before-seen Thenext-generationopticalfibercommunicationnetworkaimsto
connectionexperienceacrossawiderareafornumerousdevices112,113. achieve a Fiber to Everywhere vision118,119, featuring ultra-high
Challenges.Inordertorealizethebeyond-5Gvision,severaltechnical bandwidth(upto800G~1.6Tbpstransmissioncapacityperfiber),
challengesneedtobeaddressed.Themostcrucialoneisachieving all-optical connectivity (establishing an all-optical network with
low-latency,high-reliabilitynetworkconnectionsforcomplexchan- ultra-low power consumption and extending fibers to deeper
nel environments and providing deterministic communication indoorsettings),andanultimateexperience(zeropacketloss,no
guarantees.Thekeytothisisactivesignalprocessingthroughpre- sense of delay, and ultra-reliability). Challenges. To attain such a
dictingpotentialchangesinthechannelbasedontheperceptionof significantvision,significanttechnologicaladvancementsmustbe
the environment. This requires overturning traditional passive made in areas such as all-optical signal processing, system opti-
waveformdesignandchannelcoding,andinstead,relyingheavilyon mization,anduncertaintycontrol.Thesetechnicalchallengescan
active sensing, accurate prediction, and dynamic optimization of benefitfromnewtheories,algorithms,andsystemarchitecturesof
complex channels to systematically optimize channel capacity. To RC.Forinstance, asiliconphotonicsintegratedRCsystem,func-
address these technical challenges while maintaining a lightweight tioning as a photonic neural network, can achieve end-to-end
deployment cost, RC can play a significant role. For example, opticaldomainsignalprocessingwithnegligiblepowerconsump-
essentialmodulessuchaswaveformoptimizationanddecodingcan tionandtimedelayinprinciple,withoutrelyingonelectro-optical/
greatlybenefitfromaccurateidentificationanddynamicestimation opticalconversion.Asaresult,ithasthepotentialtobecomeakey
ofchannelstateinformationintegratedsensingandcommunication. technology in future all-optical networks. Additionally, adjusting
Thesemodulescanalsobefurtherimprovedbytransformingfrom theinternalstructureoftheopticalfibercanenabletheenhance-
responsivetopredictivechannelestimation.Finally,real-timechan- ment of capacity by searching complex and diverse structures,
nel optimization, such as using RIS techniques, would require fast which canbenefitfromtheeffectiveand automated modeling of
andadaptivecontrolofpotentiallyhigh-dimensionaldynamics.Due thechannelwithRC.Thisapproachtransformstheoriginalblack-
to its compact and lightweight network structure, rich functional boxoptimizationofthesystemintothewhite-boxoptimizationof
NatureCommunications|(2024)15:2056 11

Perspective https://doi.org/10.1038/s41467-024-45187-1
theRC’soutputlayer,likelyabletoimprovetheoptimizationeffi- Intelligentrobots
ciency. In terms of low-latency and reliability assurance at the Opportunities.Robotsarebecomingincreasinglyimportantintoday’s
optical network level, RC research can play a critical role in link informationsocietyduetotheirabilitytotakemanyforms,including
failurepredictionearlywarning,faultlocalization,anddynamical intelligentphysicalmanifestations.Oneexampleofthisislarge-scale
control.DuetothecompactdesignofRC,embeddeddevicescan commercial sweeping robots used in smart homes115, which have
perform intelligent processing tasks as a natural part of the net- replacedtraditionalmanualoperationsinvariousscenarios,improving
worksystem,withoutrequiringacentralizedpowercenter. both production efficiency and living standards. With advances in
technology,moretypesofintelligentrobotsareexpectedtoemerge
InternetofThings(IoT) over the next decade, capable of completing complicated tasks
Opportunities. In comparison to traditional communication and throughautonomousperception,calculation,optimization,andcon-
interconnectionservicesforcomputersandmobilephones,the trolincomplexenvironmentslikefailuredetection,medicaldiagnosis,
Internet of Things (IoT) caters to a wider range of devices with and search-and-rescue operations. Biological intelligence serves as
broadercoverage,posingseveralnewtechnologicalchallenges. inspirationforachievingrobotintelligence,whichreliesonthreekey
With IoT, the quantity and types of objects served are sig- elements: real-time intensive information collection and perception
nificantly higher, including smart temperature and light capabilities(madepossiblebytechnologiessuchasflexiblesensing,
control120,121, open-space noise cancellation122, air quality electronic skin, and multi-dimensional environment modeling), fast
monitoring123, among others, all of which are key features of information processing capabilities (enabled by technologies like
smart homes. Communication technologies used to realize the decision-making optimization and dynamic control), and physical
interconnection of these devices are diverse, including Blue- control capabilities (facilitated by nonlinear modeling and electro-
tooth, NFC, visible light, RFID, WiFi, ZigBee, and so on. Chal- mechanicalcontrol).Challenges.Duetophysicalconstraintssuchas
lenges. Unlike high-end devices such as computers and mobile batterycapacityanddeploymentenvironmentuncertainty,thecore
phones,avastmajorityofIoTconnecteddevicescannotrelyon modulessupportingrobotintelligenceareexpectedtobeembedded
energy-hungry integrated chip technology to achieve advanced in the physicalentity of the robot in an offline manner rather than
computingperformanceduetopowerconsumptionandvolume relyingoncloudandnetworkcapabilitiestoprovidepotentiallarge
limitations.Consequently,IoTend-sidesystemsmustutilizelow- modelcapabilities.SimilartotheIoTscenario,machinelearningthatis
power,programmabletechniquestoachieveadaptiveperception widelyreliedoninrobotintelligencemusthavethecharacteristicsof
andcomputingnecessaryforedgeintelligence.Thelightweight miniaturization,lowenergyconsumption,andeasydeployment,while
anddynamicallycontrollablenatureofsuchrequirementsmake requiring the ability to recognize, predict, calculate, and control
RCsystemsparticularlyadvantageousoverlargeAImodels.With dynamicprocesses.ThispresentsanexcellentapplicationfieldforRC
the success of domain-specific chips for audio and video pro- systemstoplayarole.InMPC,sincetheroleofRCmerelyreplacesa
cessing,thereisexpectedtobesignificantdemandforembedded linearpredictortheoverallcontrollerarchitectureremainstransparent
smartchipsintheIoTfield,whichwillopenupnewopportunities andintact.Inprinciple,itispossibletoadoptRCforgeneralcontroller
fortheapplicationofRCresearch. design beyond usage in the MPC framework, e.g., directly learning
controlrulesfromdatatogetherwith(some)priormodelknowledge.
Greendatacenters However,themainchallengewouldbetoposetheoreticalguarantees
Opportunities. Data centers havebecomean essentialinfrastructure on error and convergence neither of which have been resolved by
forthenewgenerationofinformationsocietyduetothesubstantial existingworksofRC.
increase in demand for massive computing and data storage. It is
estimatedthatby2030,globaldatacenterswillprocessatrillionGBof AIforscienceanddigitaltwins
dataeveryday,andtheirpowerconsumptionisexpectedtoaccount Opportunities.Tofullyrealizetheongoinginformationrevolution,itis
forover60%oftotalpowergeneration.However,thelargeamountof essential to rethink and reshape crucial aspects of industrial manu-
electricityconsumptionandheatemissionsrequiredtooperatethese facturing through the innovative framework of AI for science and
centershaveasignificantimpactontheenvironment.Therefore,the digitaltwins.Thisinvolvesachievingfullperceptionandprecisecon-
designanddevelopmentofnewgenerationgreendatacenterswith trolofphysicalsystemsthroughinteractionsanditerativefeedback
low energyconsumption andhighreliabilityarecrucialfor thesus- betweendigitalmodelsandentitiesinthephysicalworld.Essentially,
tainable developmentofsociety.Challenges. The realization oflow- digital twins establish a synchronous relationship between physical
energydatacentersreliesonnumeroustechnologicalbreakthroughs. systems and their digital representations. Using this synchronous
Energy consumption in data transfer accounts for a significant pro- function,simulationscanberuninthedigitalworld,andoptimized
portion, with optical modules playing a central role. Therefore, designscanrepeatedlyanditerativelybeimportedintothephysical
achieving low energy consumption requires reducing the energy system,ultimatelyleadingtooptimizationandcontrol.Forsystems
consumption of optical modules. One promising approach is to withclearandcompletephysicalmechanisms,synchronizationmod-
implementall-opticalsignalprocessingbasedontheintegratedsilicon elsthatdigitaltwinsrelyonareusuallysetsofODEs/PDEs.Forexam-
photonics on-chip RC system. Additionally, data centers comprise ple,simulatingfullthree-dimensionalturbulence,weatherforecasting,
manycomponentsthatformanextremelycomplexdynamicsystem. laser dynamics,etc.Preliminary studies suggest thatreservoir com-
Maintainingthesystem’snormaloperationattheleastpossiblecostof putingcanbeusedtoreducethecomputationalresourcesrequiredfor
energyconsumption,suchaskeepingtheoveralltemperaturestableat these expensive simulations. Arcomano et al.111 developed a low-
alow-range,canbeviewedasanoptimalcontrolproblem.Apotential resolutionglobalpredictionmodelbasedonreservoircomputingand
solutiontothisproblemisthroughdata-drivenmodelswithphysical investigated the applicability of RC in weather forecasting. They
priors,whichcombinesastructuredmodelderivedfromtheconnec- demonstratedthataparallelMLmodelbasedonRCcanpredictthe
tionrelationshipandfunctionsofphysicalequipmentanddata-driven global atmospheric state in the same grid format as the numerical
methodstobuildadynamiccontrolframework.Bymonitoringand (physics-based)globalweatherforecastmodel.Theyalsofoundthat
adjustingtheparameterconfigurationofeachmoduleofthesystemin the current version of the ML model has potential in short-term
real-time,thisframeworkcanachievetheoptimaloperatingstatusand weather forecasting. They further discovered that when full-state
energyconsumptioncost.RChasthepotentialtoplayacrucialrolein dynamics areavailable for training, RCoutperformsthe time-based
thisapproach. backpropagationthroughtime(BPTT)methodintermsofprediction
NatureCommunications|(2024)15:2056 12

Perspective https://doi.org/10.1038/s41467-024-45187-1
performanceandcapturinglong-termstatisticaldatawhilerequiring reservoir computing system ineffective. One intuitive solution is to
less training time. Challenges. Calculations of these physics-inferred adjustthephysicalparametersofthereservoirtomatchthetimescale
equationscanbechallenging.Inmorecomplexindustrialapplications, ofthecomputationalproblem.Thisposeshighrequirementsforthe
multiple coupling modules are often present, and interactions designofRCnetworkstructuresandtrainingalgorithms.Usingother
between the system and the open environment cannot be fully technologies such as super-resolution and compressive sensing to
describedbyphysicalmechanismsormathematicalfunctions.There- overcome the resolution problem of single-point measurement and
fore,itisnecessarytoconsiderfastcalculationtechniques,butalso processinginRCsystemsmaybeaviablesolution.Thesecondisthe
findwaystobuildsynchronizationmodelsfornon-white-boxcomplex real-timedataprocessingproblem:Oneofthesignificantadvantages
dynamicsystems.Mathematicalmodelingoffusionbetweenphysical ofreservoircomputingislightweightandfastcomputation.However,
mechanismsanddata-driventechniqueshasbeensignificantlydevel- inpracticalphysicalsystems,itisoftenunrealistictosampleandstore
oped in the past decade. For instance, Physics-inspired Neural Net- alargenumberofnoderesponsestoacertaininputduetolimitations
works(PINN)embedthestructureandformofphysicalequationsinto suchassamplingbandwidth,storagedepthandbandwidth,ortheir
neural network loss functions, which guides the neural network to combinations.Itissimplynotfeasibleinmanycasestoprobeasystem
approximateprovidedphysicsequationsduringparametertraining124. withalargenumberofprobes(10s–1000s)interfacedwithADcon-
Another type of physics-inspired computing system, RC, inherently verters.Inadditiontothesepracticalchallenges,hardwaredriftoften
provides an embedding method of the mechanismmodel, which is requiresregularrepetitionofcalibrationprocedures,henceitcannot
expected to provide a powerful supplement to the solver for basic beaone-ofoptimization.Furthermore,datapreprocessingandpost-
physical models of industrial simulation, focusing on offering a processingalsolimittheoverallcomputationalspeedofthephysical
dynamicmodelingframeworkforthefusionofmechanismsanddata. RC system. One approach to address this issue is to use hardware-
However, for reduced-order data, large-scale RC models may be basedreadoutinsteadofsoftware-basedreadout126–129.
unstableandmorelikelytoexhibitbiasthantheBPTTalgorithm.In Moving forward, it is crucial that we thoroughly explore the
anotherexampleofresearchonnonlinearlaserdynamics,theauthors potentialofintelligentlearningmachinesbasedondynamicalsystems.
found that RC methods have simpler training mechanisms and can Intherealmoftheoreticalandalgorithmicresearch,itisnecessaryto
reducetrainingtimecomparedtodeepneuralnetworks125.Forprac- continuouslypushtheboundariesofperformanceandofferguidance
tical problems involving complex nonlinear physical processes, we forexperimentaldesign.Reservoircomputing(RC)researchcantake
havereasontobelievethatRCmethodsmayprovideuswithsolutions root in theory and algorithms,with experiments serving as approx-
forcomputationalacceleration. imations to theoretical and algorithmic results. However, one dis-
advantage of this approach is that it can be challenging to identify
Outlook equivalent devices in experiments that can achieve the nonlinear
Insummary,althoughRChasthepotentialforlarge-scaleapplication properties of RC in theory, which can lead to reduced accuracy.
intermsoffunctions,inordertotrulysolvethetechnicalproblemsin Alternatively,researcherscanfocusonbuildingphysicalRCsystemas
theabove-mentionedvariousmajorapplications,therearestillmany theultimate goal,which requires close collaboration between theo-
key challenges in the existing RC system in various aspects. For retical and experimental teams to optimize the system jointly. This
example, in theoretical research, although the universalapproxima- approachhastheadvantageofconsideringphysicalconstraintsand
tiontheoryofRChasadvancedsignificantlyinrecentyears,mostof applicationcharacteristicswhendesigningalgorithms,makingitmore
thetheoreticalresultsfocusonexistenceproofsandlackstructural likelytoachievebettersolutionsattheimplementationlevel.Thisalso
design.Hence,thecurrentapproximationtheoryhasnotyetplayedan raisesthebarforinterdisciplinaryresearch,asparticipantswillneedto
important guiding role in RC network architecture design, training possesscross-disciplinarycommunicationskillsandknowledge,along
methods, etc., nor can it quantitatively evaluate the approximation with an openness towards multi-module complex coupling
potentialofaspecificRCschemefordynamicsystemsortimeseries. optimization.
AnimportantreasontofurtheradvancethemathematicaltheoryofRC Looking ahead, unlocking the full potential of RC and neuro-
isfordata-drivencontrolapplications.Inmostofthoseapplications, morphiccomputingingeneraliscriticalyetchallenging.Infact,this
rigoroustheoryoncontrolerrorandconvergencearenecessaryfor goesbeyondjustputtingoutopen-sourcecodesorsolveafewspecific
thecorrespondingcontrollertobeconsideredusableinanindustrial problems.Innovativeideasandinterdisciplinaryresearchformatsare
setting.However,sofarverylittleworkhasbeendonetoaddressthese much needed. As concrete suggestions, researchers of the applied
important problems. As for algorithmic challenges, most industrial mathematicsandnonlineardynamicscommunitieswhohavebeenthe
applicationsdonotrequireauniversalapproximator,butinthesame main players in RC will need to get close(r) to the mainstream AI
field,theapproximationmodelneedstobegeneralizable.ExistingRC applicationsandtrytodevelopnext-generationRCsystemstocom-
research has very little exploration in domain-specific architecture pete in these scenarios where the value of application has been
optimization.Problemsintheindustrialfieldaredividedintoscenarios establishedandrecognizedbytheindustry.Agoodstartingpointcan
andcategories.Therefore,itisimportanttoconstructgeneral-purpose beopen-sourcetasksanddatasetssuchasKaggle,andmoregenerally
RC models possibly by means of architecture search. In addition, to directly partner with industrial research labs to put RC into real
leavingasidethepracticalityofRCforthetimebeing,pastresearchhas applications.Ontheotherhand,raisingawarenessofthe(potential)
turned its advantages into constraints, such as small size, simple utilityofRCrequiresattractinginterestfromresearchersanddecision-
training, and so on. However, how strong is RC’s learning ability makerswhoaretraditionallyoutsideofthefield.Forinstance,themed
(whether there is an RC architecture that can compare with GPT’s conferencesandworkshopsmaybeorganizedtofostersuchdiscus-
ability),itisstillunknown. sions among scientists and researchers from diverse fields across
Attheexperimentallevel,therearestillsomegapswhenmapping academiaandindustry.Despitethemanychallenges,withpersistence
RC models to physical systems. The first is timescale problem of andinnovationsanewandfutureparadigmofintelligentlearningand
physicalsubstrateRC:Matchingthetimescalesbetweenthecompu- computing may possibly emerge from the works of RC and neuro-
tationalchallengeandtheinternaldynamicsofthephysicalRCsub- morphiccomputing.
strate is a key issue in reservoir computing. If the timescale of the
problemismuchfasterthantheresponsetimeofthephysicalsystem, References
theresponseofthereservoirwillbetoosmallorthefadingmemoryof 1. Graves,A.,Mohamed,A.R.&Hinton,G.Speechrecognitionwith
the reservoir will not be properly utilized, rendering the physical deeprecurrentneuralnetworks.InIEEEInternationalConference
NatureCommunications|(2024)15:2056 13

Perspective https://doi.org/10.1038/s41467-024-45187-1
onAcoustics,SpeechandSignalProcessing,6645–6649 21. Pathak,J.,Hunt,B.,Girvan,M.,Lu,Z.&Ott,E.Model-freepredic-
(IEEE,2013). tionoflargespatiotemporallychaoticsystemsfromdata:A
2. LeCun,Y.,Bengio,Y.&Hinton,G.E.Deeplearning.Nature521, reservoircomputingapproach.Phys.Rev.Lett.120,024102
436–444(2015). (2018).ThispaperproposesaparallelRCarchitecturetolearnthe
3. He,K.,Zhang,X.,Ren,S.&Sun,J.Deepresiduallearningforimage behaviorofKuramoto-Sivashinsky(KS)equations.Theworkshows
recognition.InIEEEConferenceonComputerVisionandPattern theexcitingpotentialofRCinlearningthecomputationalbeha-
Recognition(CVPR),770–778(IEEE,2016). viorandstateevolutionofPDEs.
4. Silver,D.etal.Masteringthegameofgowithdeepneuralnet- 22. Vlachas,P.R.etal.Backpropagationalgorithmsandreservoir
worksandtreesearch.Nature529,484–489(2016). computinginrecurrentneuralnetworksfortheforecastingof
5. Krizhevsky,A.,Sutskever,I.&Hinton,G.E.Imagenetclassification complexspatiotemporaldynamics.NeuralNetw.126,
withdeepconvolutionalneuralnetworks.Commun.ACM60, 191–217(2020).
84–90(2017). 23. Bianchi,F.M.,Scardapane,S.,Løkse,S.&Jenssen,R.Reservoir
6. Jumper,J.etal.Highlyaccurateproteinstructurepredictionwith computingapproachesforrepresentationandclassificationof
alphafold.Nature596,583–589(2021). multivariatetimeseries.IEEETrans.NeuralNetw.Learn.Syst.32,
7. Brown,T.etal.Languagemodelsarefew-shotlearners.NeurIPS 2169–2179(2020).
33,1877–1901(2020). 24. Gauthier,D.J.,Bollt,E.,Griffith,A.&Barbosa,W.A.Nextgenera-
8. Khan,A.,Sohail,A.,Zahoora,U.&Qureshi,A.S.Asurveyofthe tionreservoircomputing.Nat.Commun.12,1–8(2021).Thiswork
recentarchitecturesofdeepconvolutionalneuralnetworks.Artif. revealsanintriguinglinkbetweentraditionalRCandregression
Intell.Rev.53,5455–5516(2020). methodsandinparticularshowsthatnonlinearvectorauto-
9. Schuman,C.D.etal.Opportunitiesforneuromorphiccomputing regression(NVAR)canequivalentlyrepresentRCwhilerequiring
algorithmsandapplications.Nat.Comput.Sci2,10–19(2022). fewerparameterstotune,leadingtothedevelopmentofso-called
10. Christensen,D.V.etal.2022roadmaponneuromorphiccomputing next-generationRC,showntooutperformtraditionalRCwithless
andengineering.Neuromorph.Comput.Eng.2,022501(2022). dataandhigherefficiency,pushingforwardasignificantstepfor
11. Jaeger,H.The“echostate”approachtoanalysingandtraining constructinganinterpretablemachinelearning.
recurrentneuralnetworks-withanerratumnote.Bonn,Germany: 25. Joy,H.,Mattheakis,M.&Protopapas,P.Rctorch:apytorchreser-
GermanNat.Res.CenterforInf.Technol.GMDTech.Rep.148,13 voircomputingpackagewithautomatedhyper-parameteropti-
(2001).Thefirstpaperdevelopingtheconceptandframeworkof mization.Preprintathttps://doi.org/10.48550/arXiv.2207.
echostatenetworks,e.g.reservoircomputing.Thepaperprovides 05870(2022).
propositionsonhowtoconstructESNsandhowtotrainthem.The 26. Griffith,A.,Pomerance,A.&Gauthier,D.J.Forecastingchaotic
paperalsoshowsthattheESNisabletolearnandpredictchaotic systemswithverylowconnectivityreservoircomputers.Chaos
timeseries(Mackey-Glassequations). 29,123108(2019).
12. Maass,W.,Natschläger,T.&Markram,H.Real-timecomputing 27. Yperman,J.&Becker,T.Bayesianoptimizationofhyper-
withoutstablestates:Anewframeworkforneuralcomputation parametersinreservoircomputing.Preprintathttps://doi.org/10.
basedonperturbations.NeuralComput.14,2531–2560(2002). 48550/arXiv.1611.05193(2016).
Thefirstpaperproposingtheideaofliquidstatemachines.The 28. Appeltant,L.etal.Informationprocessingusingasingledyna-
modelisabletolearnfromabundantperturbedstatessoasto micalnodeascomplexsystem.Nat.Commun.2,1–6(2011).
learnvarioussequences,andcanalsofulfillreal-timesignalpro- 29. Paquot,Y.etal.Optoelectronicreservoircomputing.Sci.Rep.2,
cessingfortime-varyinginputs.Thispaperdemonstratesthat 1–6(2012).
LSMscanbeusedforlearningtaskssuchasspoken-digit 30. Larger,L.etal.High-speedphotonicreservoircomputingusinga
recognition. time-delay-basedarchitecture:Millionwordspersecondclassifi-
13. Verstraeten,D.,Schrauwen,B.,D’Haene,M.&Stroobandt,D.The cation.Phys.Rev.X7,011015(2017).
unifiedreservoircomputingconceptanditsdigitalhardware 31. Dong,J.,Rafayelyan,M.,Krzakala,F.&Gigan,S.Opticalreservoir
implementations.InProceedingsofthe2006EPFLLATSISSym- computingusingmultiplelightscatteringforchaoticsystems
posium,139–140(EPFL,Lausanne,2006). prediction.IEEEJ.Sel.Top.QuantumElectron.26,1–12(2019).
14. Zhu,Q.,Ma,H.&Lin,W.Detectingunstableperiodicorbitsbased 32. Rafayelyan,M.,Dong,J.,Tan,Y.,Krzakala,F.&Gigan,S.Large-
onlyontimeseries:Whenadaptivedelayedfeedbackcontrol scaleopticalreservoircomputingforspatiotemporalchaotic
meetsreservoircomputing.Chaos29,093125(2019). systemsprediction.Phys.Rev.X10,041037(2020).
15. Bertschinger,N.&Natschläger,T.Real-timecomputationatthe 33. Du,C.etal.Reservoircomputingusingdynamicmemristorsfor
edgeofchaosinrecurrentneuralnetworks.NeuralComput.16, temporalinformationprocessing.Nat.Commun.8,1–10(2017).
1413–1436(2004). TheworkdevelopsaphysicalRCsystembasedonmemristor
16. Rodan,A.&Tino,P.Minimumcomplexityechostatenetwork.IEEE arrays,findingthatsuchasystemisabletoperformwellinrea-
Trans.NeuralNetw.22,131–144(2010). lizinghandwrittendigitrecognitionandsolvingasecond-order
17. Verzelli,P.,Alippi,C.,Livi,L.&Tino,P.Input-to-staterepresenta- nonlineardynamictaskswithlessthan100reservoirnodes.
tioninlinearreservoirsdynamics.IEEETrans.NeuralNetw.Learn. 34. Moon,J.etal.Temporaldataclassificationandforecastingusinga
Syst.33,4598–4609(2021). memristor-basedreservoircomputingsystem.Nat.Electron.2,
18. Gallicchio,C.,Micheli,A.&Pedrelli,L.Deepreservoircomputing: 480–487(2019).
Acriticalexperimentalanalysis.Neurocomputing268, 35. Zhong,Y.etal.Dynamicmemristor-basedreservoircomputingfor
87–99(2017). high-efficiencytemporalsignalprocessing.Nat.Commun.12,
19. Gallicchio,C.,Micheli,A.&Pedrelli,L.Designofdeepechostate 1–9(2021).
networks.NeuralNetw.108,33–47(2018). 36. Sun,L.etal.In-sensorreservoircomputingforlanguagelearning
20. Gallicchio,C.&Scardapane,S.Deeprandomizedneuralnet- viatwo-dimensionalmemristors.Sci.Adv.7,eabg1455(2021).
works.InRecentTrendsinLearningFromData:Tutorialsfromthe 37. Lin,W.&Chen,G.Largememorycapacityinchaoticartificial
INNSBigDataandDeepLearningConference,43–68(Springer neuralnetworks:Aviewoftheanti-integrablelimit.IEEETrans.
Cham,Switzerland,2020). NeuralNetw.20,1340–1351(2009).
NatureCommunications|(2024)15:2056 14

Perspective https://doi.org/10.1038/s41467-024-45187-1
38. Silva,N.A.,Ferreira,T.D.&Guerreiro,A.Reservoircomputingwith 61. Cucchi,M.etal.Reservoircomputingwithbiocompatibleorganic
solitons.NewJ.Phys.23,023013(2021). electrochemicalnetworksforbrain-inspiredbiosignalclassifica-
39. Ghosh,S.,Opala,A.,Matuszewski,M.,Paterek,T.&Liew,T.C. tion.Sci.Adv.7,eabh0693(2021).
Quantumreservoirprocessing.npjQuantumInf.5,1–6(2019). 62. Rowlands,G.E.etal.Reservoircomputingwithsuperconducting
Proposedaplatformforquantuminformationprocessingdevel- electronics.Preprintathttps://doi.org/10.48550/arXiv.2103.
opedontheprincipleofreservoircomputing. 02522(2021).
40. Govia,L.C.G.,Ribeill,G.J.,Rowlands,G.E.,Krovi,H.K.&Ohki,T. 63. Verstraeten,D.,Schrauwen,B.&Stroobandt,D.Reservoircom-
A.Quantumreservoircomputingwithasinglenonlinearoscillator. putingwithstochasticbitstreamneurons.InProceedingsofthe
Phys.Rev.Res.3,013077(2021). 16thAnnualProriscWorkshop,454–459(2005).https://doi.org/
41. Buehner,M.&Young,P.Atighterboundfortheechostate https://biblio.ugent.be/publication/336133.
property.IEEETrans.NeuralNetw.17,820–824(2006). 64. Schürmann,F.,Meier,K.&Schemmel,J.Edgeofchaoscompu-
42. Jaeger,H.ShortTermMemoryinEchoStateNetworks.Technical tationinmixed-modevlsi-ahardliquid.NeurIPS,17,(NIPS,2004).
Report152(GMD,Berlin,2001). 65. Torrejon,J.etal.Neuromorphiccomputingwithnanoscalespin-
43. Duan,X.Y.etal.Embeddingtheoryofreservoircomputingand tronicoscillators.Nature547,428–431(2017).Firstdemonstration
reducingreservoirnetworkusingtimedelays.Phys.Rev.Res.5, ofRCimplementationusingaspintronicoscillator,opensupa
L022041(2023). routetorealizinglarge-scaleneuralnetworksusingmagnetization
44. Boyd,S.&Chua,L.Fadingmemoryandtheproblemofapprox- dynamics.
imatingnonlinearoperatorswithvolterraseries.IEEETrans.Cir- 66. Vandoorne,K.etal.Experimentaldemonstrationofreservoir
cuitsSyst.32,1150–1161(1985). computingonasiliconphotonicschip.Nat.Commun.5,1–6
45. Grigoryeva,L.&Ortega,J.P.Echostatenetworksareuniversal. (2014).Firstdemonstrationofon-chipintegratedphotonicreser-
NeuralNetw.108,495–508(2018). voirneuralnetwork,pavesthewayforthehighdensityandhigh
46. Gonon,L.&Ortega,J.P.Fadingmemoryechostatenetworksare speedsphotonicRCarchitecture.
universal.NeuralNetw.138,10–13(2021). 67. Larger,L.etal.Photonicinformationprocessingbeyondturing:an
47. Gonon,L.&Ortega,J.P.Reservoircomputinguniversalitywith optoelectronicimplementationofreservoircomputing.Opt.
stochasticinputs.IEEETrans.NeuralNetw.Learn.Syst.31, Express20,3241–3249(2012).Thispaperproposedoptical-based
100–112(2019). time-delayfeedbackRCarchitecturewithasinglenonlinear
48. Hart,A.,Hook,J.&Dawes,J.Embeddingandapproximationthe- optoelectronichardware.TheexperimentshowsthattheRC
oremsforechostatenetworks.NeuralNetw.128,234–247(2020). performswellinspoken-digitrecognitionandone-time-step
49. Hart,A.G.,Hook,J.L.&Dawes,J.H.Echostatenetworkstrainedby predictiontasks.
tikhonovleastsquaresarel2(μ)approximatorsofergodicdyna- 68. Duport,F.,Schneider,B.,Smerieri,A.,Haelterman,M.&Massar,S.
micalsystems.PhysicaDNonlinearPhenomena421,132882(2021). All-opticalreservoircomputing.Opt.Express20,22783–22795
50. Gonon,L.,Grigoryeva,L.&Ortega,J.P.Riskboundsforreservoir (2012).ThefirstpapertodevelopRCsystemwithafiber-basedall-
computing.J.Mach.Learn.Res.21,9684–9744(2020). opticalarchitecture.TheexperimentsshowthattheRCcanbe
51. Bollt,E.Onexplainingthesurprisingsuccessofreservoircom- utilizedinchannelequalizationandradarsignalpredictiontasks.
putingforecasterofchaos?theuniversalmachinelearning 69. Brunner,D.&Fischer,I.Reconfigurablesemiconductorlaser
dynamicalsystemwithcontrasttovaranddmd.Chaos31, networksbasedondiffractivecoupling.Opt.Lett.40,
013108(2021). 3854–3857(2015).
52. Krishnakumar,A.,Ogras,U.,Marculescu,R.,Kishinevsky,M.& 70. Gan,V.M.,Liang,Y.,Li,L.,Liu,L.&Yi,Y.Acost-efficientdigitalesn
Mudge,T.Domain-specificarchitectures:Researchproblemsand architectureonfpgaforofdmsymboldetection.ACMJ.Emerg.
promisingapproaches.ACMTrans.Embed.Comput.Syst.22, Technol.Comput.Syst.17,1–15(2021).
1–26(2023). 71. Elbedwehy,A.N.,El-Mohandes,A.M.,Elnakib,A.&Abou-Elsoud,
53. Subramoney,A.,Scherr,F.&Maass,W.Reservoirslearntolearn. M.E.Fpga-basedreservoircomputingsystemforecgdenoising.
ReservoirComputing:Theory,PhysicalImplementations,and Microprocess.Microsyst.91,104549(2022).
Applications,59–76(SpringerSingapore,2021). 72. Lin,C.,Liang,Y.&Yi,Y.Fpga-basedreservoircomputingwith
54. Tanaka,G.etal.Recentadvancesinphysicalreservoircomputing: optimizedreservoirnodearchitecture.In23rdInternationalSym-
Areview.NeuralNetw.115,100–123(2019). posiumonQualityElectronicDesign(ISQED),1–6(IEEE,2022).
55. Jiang,W.etal.Physicalreservoircomputingusingmagneticsky- 73. Bai,K.&Yi,Y.Dfr:Anenergy-efficientanalogdelayfeedback
rmionmemristorandspintorquenano-oscillator.Appl.Phys.Lett. reservoircomputingsystemforbrain-inspiredcomputing.ACMJ.
115,192403(2019). Emerg.Technol.Comput.Syst.14,1–22(2018).
56. Coulombe,J.C.,York,M.C.&Sylvestre,J.Computingwithnet- 74. Petre,P.&Cruz-Albrecht,J.Neuromorphicmixed-signalcircuitry
worksofnonlinearmechanicaloscillators.PLOSONE12, forasynchronouspulseprocessing.InIEEEInternationalCon-
e0178663(2017). ferenceonRebootingComputer,1–4(IEEE,2016).
57. Larger,L.,Goedgebuer,J.P.&Udaltsov,V.Ikeda-basednonlinear 75. Nowshin,F.,Zhang,Y.,Liu,L.&Yi,Y.Recentadvancesinreservoir
delayeddynamicsforapplicationtosecureopticaltransmission computingwithafocusonelectronicreservoirs.InInternational
systemsusingchaos.C.R.Phys.5,669–681(2004). GreenandSustainableComputingWorkshops,1–8(IEEE,2020).
58. Brunner,D.,Soriano,M.C.,Mirasso,C.R.&Fischer,I.Parallel 76. Soriano,M.C.etal.Delay-basedreservoircomputing:noise
photonicinformationprocessingatgigabyteperseconddata effectsinacombinedanaloganddigitalimplementation.IEEE
ratesusingtransientstates.Nat.Commun.4,1364(2013). Trans.NeuralNetw.Learn.Syst.26,388–393(2014).
59. Katayama,Y.,Yamane,T.,Nakano,D.,Nakane,R.&Tanaka,G. 77. Marinella,M.J.&Agarwal,S.Efficientreservoircomputingwith
Wave-basedneuromorphiccomputingframeworkforbrain-like memristors.Nat.Electron.2,437–438(2019).
energyefficiencyandintegration.IEEETrans.Nanotechnol.15, 78. Sun,W.etal.3dreservoircomputingwithhighareaefficiency
762–769(2016). (5.12tops/mm2)implementedby3ddynamicmemristorarrayfor
60. Dion,G.,Mejaouri,S.&Sylvestre,J.Reservoircomputingwitha temporalsignalprocessing.InIEEESymposiumonVLSITechnol-
singledelay-couplednon-linearmechanicaloscillator.J.Appl. ogyandCircuits(VLSITechnologyandCircuits),222–223
Phys.124,152132(2018). (IEEE,2022).
NatureCommunications|(2024)15:2056 15

Perspective https://doi.org/10.1038/s41467-024-45187-1
79. Allwood,D.A.etal.Aperspectiveonphysicalreservoircomputing AAAI21Workshop:AIforUrbanMobility,(2021).https://aaai.org/
withnanomagneticdevices.Appl.Phys.Lett.122,040501(2023). conference/aaai/aaai-21/ws21workshops/.
80. Shen,Y.etal.Deeplearningwithcoherentnanophotoniccircuits. 101. Yamane,T.etal.Applicationidentificationofnetworktrafficby
Nat.Photonics11,441–446(2017). reservoircomputing.InInternationalConferenceonNeuralInfor-
81. VanderSande,G.,Brunner,D.&Soriano,M.C.Advancesin mationProcessing,389–396(SpringerCham,2019).
photonicreservoircomputing.Nanophotonics6,561–576(2017). 102. Ando,H.&Chang,H.Roadtrafficreservoircomputing.Preprintat
82. Maass,W.,Natschläger,T.&Markram,H.Amodelforreal-time https://doi.org/10.48550/arXiv.1912.00554(2019).
computationingenericneuralmicrocircuits.NeurIPS15 103. Wang,J.,Niu,T.,Lu,H.,Yang,W.&Du,P.Anovelframeworkof
(NIPS,2002). reservoircomputingfordeterministicandprobabilisticwind
83. Verstraeten,D.,Schrauwen,B.,Stroobandt,D.&VanCampenh- powerforecasting.IEEETrans.Sustain.Energy11,337–349(2019).
out,J.Isolatedwordrecognitionwiththeliquidstatemachine:a 104. Joshi,P.&Maass,W.Movementgenerationandcontrolwith
casestudy.Inf.Process.Lett.95,521–528(2005). genericneuralmicrocircuits.InInternationalWorkshoponBiolo-
84. Verstraeten,D.,Schrauwen,B.&Stroobandt,D.Reservoir-based gicallyInspiredApproachestoAdvancedInformationTechnology,
techniquesforspeechrecognition.InIEEEInternationalJoint 258–273(Springer,2004).
ConferenceonNeuralNetworkProceedings,1050–1053 105. Burgsteiner,H.Trainingnetworksofbiologicalrealisticspiking
(IEEE,2006). neuronsforreal-timerobotcontrol.InProceedingsofthe9th
85. Jalalvand,A.,VanWallendael,G.&VandeWalle,R.Real-time internationalconferenceonengineeringapplicationsofneural
reservoircomputingnetwork-basedsystemsfordetectiontasks networks,129–136(2005).https://users.abo.fi/abulsari/
onvisualcontents.In7thInternationalConferenceonComputa- EANN.html.
tionalIntelligence,CommunicationSystemsandNetworks, 106. Burgsteiner,H.,Kröll,M.,Leopold,A.&Steinbauer,G.Movement
146–151(IEEE,2015). predictionfromreal-worldimagesusingaliquidstatemachine.In
86. Nakajima,M.,Tanaka,K.&Hashimoto,T.Scalablereservoircom- InnovationsinAppliedArtificialIntelligence:18thInternational
putingoncoherentlinearphotonicprocessor.Commun.Phys.4, ConferenceonIndustrialandEngineeringApplicationsofArtificial
20(2021). IntelligenceandExpertSystems,121–130(Springer,2005).
87. Cao,J.etal.Emergingdynamicmemristorsforneuromorphic 107. Schwedersky,B.B.,Flesch,R.C.C.,Dangui,H.A.S.&Iervolino,L.
reservoircomputing.Nanoscale14,289–298(2022). A.Practicalnonlinearmodelpredictivecontrolusinganechostate
88. Jaeger,H.&Haas,H.Harnessingnonlinearity:Predictingchaotic networkmodel.InIEEEInternationalJointConferenceonNeural
systemsandsavingenergyinwirelesscommunication.Science Networks(IJCNN),1–8(IEEE,2018).
304,78–80(2004). 108. Canaday,D.,Pomerance,A.&Gauthier,D.J.Model-freecontrolof
89. Nguimdo,R.M.&Erneux,T.Enhancedperformancesofaphotonic dynamicalsystemswithdeepreservoircomputing.J.Phys.
reservoircomputerbasedonasingledelayedquantumcascade Complexity2,035025(2021).
laser.Opt.Lett.44,49–52(2019). 109. Baldini,P.Reservoircomputinginrobotics:areview.Preprintat
90. Argyris,A.,Bueno,J.&Fischer,I.Photonicmachinelearning https://doi.org/10.48550/arXiv.2206.11222(2022).
implementationforsignalrecoveryinopticalcommunications. 110. Arcomano,T.,Szunyogh,I.,Wikner,A.,Hunt,B.R.&Ott,E.A
Sci.Rep.8,1–13(2018). hybridatmosphericmodelincorporatingmachinelearning
91. Argyris,A.etal.Comparisonofphotonicreservoircomputing cancapturedynamicalprocessesnotcapturedbyitsphysics-
systemsforfibertransmissionequalization.IEEEJ.Sel.Top. basedcomponent.Geophys.Res.Lett.50,e2022GL102649
QuantumElectron.26,1–9(2019). (2023).
92. Sackesyn,S.,Ma,C.,Dambre,J.&Bienstman,P.Experimental 111. Arcomano,T.etal.Amachinelearning-basedglobalatmospheric
realizationofintegratedphotonicreservoircomputingfornon- forecastmodel.Geophys.Res.Lett.47,e2020GL087776(2020).
linearfiberdistortioncompensation.Opt.Express29, Thisworkextendsthe“parallelRC”frameworkintheapplication
30991–30997(2021). ofweatherforecasting,suggestinggreatpotentialofRCinchal-
93. Sozos,K.etal.High-speedphotonicneuromorphiccomputing lengingreal-worldscenariosatafractionofthecostofdeep
usingrecurrentopticalspectrumslicingneuralnetworks.Comms. neuralnetworks.
Eng.1,24(2022). 112. Latva-Aho,M.&Leppänen,K.Keydriversandresearchchallenges
94. Soh,H.&Demiris,Y.Iterativetemporallearningandprediction for6gubiquitouswirelessintelligence.https://urn.fi/URN:ISBN:
withthesparseonlineechostategaussianprocess.InInterna- 9789526223544(2019).
tionalJointConferenceonNeuralNetworks(IJCNN),1–8 113. Rong,B.6G:TheNextHorizon:FromConnectedPeopleand
(IEEE,2012). ThingstoConnectedIntelligence.IEEEWirel.Commun.28,
95. Kim,J.Z.,Lu,Z.,Nozari,E.,Pappas,G.J.&Bassett,D.S.Teaching 8–8(2021).
recurrentneuralnetworkstoinferglobaltemporalstructurefrom 114. Mytton,D.&Ashtine,M.Sourcesofdatacenterenergyestimates:
localexamples.Nat.Mach.Intell.3,316–323(2021). Acomprehensivereview.Joule6,2032–2056(2022).
96. Li,X.etal.Tippingpointdetectionusingreservoircomputing. 115. Jung,J.H.&Lim,D.G.Industrialrobots,employmentgrowth,and
Research6,0174(2023). laborcost:Asimultaneousequationanalysis.Technol.Forecast.
97. Jaeger,H.Adaptivenonlinearsystemidentificationwithecho Soc.Change159,120202(2020).
statenetworks.InNeurIPS,15(NIPS,2002). 116. Boschert,S.&Rosen,R.Digitaltwin-thesimulationaspect.In
98. Goudarzi,A.,Banda,P.,Lakin,M.R.,Teuscher,C.&Stefanovic,D. MechatronicFutures:ChallengesandSolutionsforMechatronic
Acomparativestudyofreservoircomputingfortemporalsignal SystemsandTheirDesignersPage59–74(SpringerCham,Swit-
processing.Preprintathttps://doi.org/10.48550/arXiv.1401. zerland,2016).
2224(2014). 117. Kao,C.K.Nobellecture:Sandfromcenturiespast:Sendfuture
99. Walleshauser,B.&Bollt,E.Predictingseasurfacetemperatures voicesfast.Rev.Mod.Phys.82,2299(2010).
withcoupledreservoircomputers.NonlinearProcess.Geophys. 118. Hillerkuss,D.,Brunner,M.,Jun,Z.&Zhicheng,Y.Avisiontowards
29,255–264(2022). f5gadvancedandf6g.In13thInternationalSymposiumonCom-
100. Okamoto,T.etal.Predictingtrafficbreakdowninurbanexpress- municationSystems,NetworksandDigitalSignalProcessing
waysbasedonsimplifiedreservoircomputing.InProceedingsof (CSNDSP)483–487(IEEE,2022).
NatureCommunications|(2024)15:2056 16

Perspective https://doi.org/10.1038/s41467-024-45187-1
119. Liu,X.OpticalCommunicationsinthe5GEra(AcademicPress, detection.InProceedingsoftheAAAIConferenceonArtificial
Cambridge,2021). Intelligence34,1266–1273(AAAI,2020).
120. Liu,Q.,Ma,Y.,Alhussein,M.,Zhang,Y.&Peng,L.Greendata 139. Zhou,Z.,Liu,L.&Xu,J.Harnessingtensorstructures-multi-mode
centerwithiotsensingandcloud-assistedsmarttemperature reservoircomputinganditsapplicationinmassivemimo.IEEE
controlsystem.Comput.Netw.101,104–112(2016). Trans.Wirel.Commun.21,8120–8133(2022).
121. Magno,M.,Polonelli,T.,Benini,L.&Popovici,E.Alowcost,highly 140. Wanshi,C.etal.5g-advancedtowards6g:Past,present,and
scalablewirelesssensornetworksolutiontoachievesmartled future.IEEEJ.Sel.AreasCommun.41,1592–1619(2023).
lightcontrolforgreenbuildings.IEEESens.J.15,2963–2973 141. Möller,T.etal.Distributedfibreopticsensingforsinkhole
(2014). earlywarning:experimentalstudy.Géotechniqu73,
122. Shen,S.,Roy,N.,Guan,J.,Hassanieh,H.&Choudhury,R.R.Mute: 701–715(2023).
bringingiottonoisecancellation.InProceedingsofthe2018 142. Liu,X.etal.Ai-basedmodelingandmonitoringtechniques
ConferenceoftheACMSpecialInterestGrouponDataCommu- forfutureintelligentelasticopticalnetworks.Appl.Sci.10,
nication,282–296(ACM,2018). 363(2020).
123. Mokrani,H.,Lounas,R.,Bennai,M.T.,Salhi,D.E.&Djerbi,R.Air 143. Saif,W.S.,Esmail,M.A.,Ragheb,A.M.,Alshawi,T.A.&Alshebeili,
qualitymonitoringusingiot:Asurvey.InIEEEInternationalCon- S.A.Machinelearningtechniquesforopticalperformancemon-
ferenceonSmartInternetofThings(SmartIoT),127–134(IEEE,2019). itoringandmodulationformatidentification:Asurvey.IEEE
124. Raissi,M.,Perdikaris,P.&Karniadakis,G.E.Physics-informed Commun.Surv.Tutor.22,2839–2882(2020).
neuralnetworks:Adeeplearningframeworkforsolvingforward 144. Song,H.,Bai,J.,Yi,Y.,Wu,J.&Liu,L.Artificialintelligenceenabled
andinverseproblemsinvolvingnonlinearpartialdifferential internetofthings:Networkarchitectureandspectrumaccess.
equations.J.Comput.Phys.378,686–707(2019). IEEEComput.Intell.Mag.15,44–51(2020).
125. Amil,P.,Soriano,M.C.&Masoller,C.Machinelearningalgorithms 145. Nyman,J.,Caluwaerts,K.,Waegeman,T.&Schrauwen,B.System
forpredictingtheamplitudeofchaoticlaserpulses.Chaos29, modelingforactivenoisecontrolwithreservoircomputing.In9th
113111(2019). IASTEDInternationalConferenceonSignalProcessing,Pattern
126. Antonik,P.etal.Onlinetrainingofanopto-electronicreservoir Recognition,andApplications,162–167(IASTED,2012).
computerappliedtoreal-timechannelequalization.IEEETrans. 146. Hamedani,K.etal.Detectingdynamicattacksinsmartgridsusing
NeuralNetw.Learn.Syst.28,2686–2698(2016). reservoircomputing:Aspikingdelayedfeedbackreservoirbased
127. Porte,X.etal.Acomplete,parallelandautonomousphotonic approach.IEEETrans.Emerg.Top.Comput.Intell.4,
neuralnetworkinasemiconductormultimodelaser.J.Phys. 253–264(2019).
Photon.3,024017(2021). 147. Patel,Y.S.,Jaiswal,R.&Misra,R.Deeplearning-basedmultivariate
128. Gholami,A.,Yao,Z.,Kim,S.,Mahoney,M.W.,andKeutzer,K.Ai resourceutilizationpredictionforhotspotsandcoldspotsmiti-
andmemorywall.RiseLabMediumPost,UniversityofCalifonia gationingreenclouddatacenters.J.Supercomput.78,
Berkeley.https://medium.com/riselab/ai-and-memory-wall- 5806–5855(2022).
2cb4265cb0b8(2021). 148. Antonelo,E.A.&Schrauwen,B.Onlearningnavigationbehaviors
129. Dai,Y.,Yamamoto,H.,Sakuraba,M.&Sato,S.Computational forsmallmobilerobotswithreservoircomputingarchitectures.
efficiencyofamodularreservoirnetworkforimagerecognition. IEEETrans.NeuralNetw.Learn.Syst.26,763–780(2014).
Front.Comput.Neurosci.15,594337(2021). 149. Dragone,M.,Gallicchio,C.,Guzman,R.&Micheli,A.RSS-based
130. Komkov,H.B.ReservoirComputingwithBooleanLogicNetwork robotlocalizationincriticalenvironmentsusingreservoircom-
Circuits.Doctoraldissertation,(UniversityofMaryland,College puting.InThe24thEuropeanSymposiumonArtificialNeuralNet-
Park,2021). works(ESANN,2016).
131. Zhang,Y.,Li,P.,Jin,Y.&Choe,Y.Adigitalliquidstatemachinewith 150. Sumioka,H.,Nakajima,K.,Sakai,K.,Minato,T.&Shiomi,M.
biologicallyinspiredlearninganditsapplicationtospeech Wearabletactilesensorsuitfornaturalbodydynamicsextraction:
recognition.IEEETrans.NeuralNetw.Learn.Syst.26, casestudyonposturepredictionbasedonphysicalreservoir
2635–2649(2015). computing.InIEEE/RSJInternationalConferenceonIntelligent
132. Dai,Z.etal.Ascalablesmall-footprinttime-space-pipelined RobotsandSystems(IROS),9504–9511(IEEE,2021).
architectureforreservoircomputing.IEEETrans.CircuitsSyst.II: 151. Wang,K.etal.Areviewofmicrosoftacademicservicesforscience
ExpressBriefs70,3069–3073(2023). ofsciencestudies.Front.BigData2,45(2019).
133. Bai,K.,Liu,L.&Yi,Y.Spatial-temporalhybridneuralnetworkwith 152. Smolensky,P.,McCoy,R.,Fernandez,R.,Goldrick,M.&Gao,J.
computing-in-memoryarchitecture.IEEETrans.CircuitsSyst.I: Neurocompositionalcomputing:Fromthecentralparadoxof
Regul.Pap.68,2850–2862(2021). cognitiontoanewgenerationofaisystems.AIMag.43,
134. Watt,S.,Kostylev,M.,Ustinov,A.B.&Kalinikos,B.A.Implement- 308–322(2022).
ingamagnonicreservoircomputermodelbasedontime-delay 153. Callaway,E.‘itwillchangeeverything’:Deepmind’saimakes
multiplexing.Phys.Rev.Appl.15,064060(2021). giganticleapinsolvingproteinstructures.Nature588,
135. Qin,J.,Zhao,Q.,Yin,H.,Jin,Y.&Liu,C.Numericalsimulationand 203–205(2020).
experimentonopticalpacketheaderrecognitionutilizingreser- 154. Callaway,E.Theentireproteinuniverse’:Aipredictsshapeof
voircomputingbasedonoptoelectronicfeedback.IEEEPhotonics nearlyeveryknownprotein.Nature608,15–16(2022).
J.9,1–11(2017). 155. Lee,P.,Bubeck,S.&Petro,J.Benefits,limits,andrisksofgpt-4
136. Susandhika,M.Acomprehensivereviewandcomparativeanalysis asanaichatbotformedicine.N.Engl.J.Med.388,
of5gand6gbasedmimochannelestimationtechniques.In 1233–1239(2023).
InternationalConferenceonRecentTrendsinElectronicsand 156. Hu,Z.,Jagtap,A.D.,Karniadakis,G.E.&Kawaguchi,K.Augmented
Communication(ICRTEC),1–8(IEEE,2023). physics-informedneuralnetworks(apinns):Agatingnetwork-
137. Chang,H.H.,Liu,L.&Yi,Y.Deepechostateq-network(deqn)and basedsoftdomaindecompositionmethodology.Eng.Appl.Artif.
itsapplicationindynamicspectrumsharingfor5gandbeyond. Intell.126,107183(2023).
IEEETrans.NeuralNetw.Learn.Syst.33,929–939(2020). 157. Kashinath,K.etal.Physics-informedmachinelearning:casestu-
138. Zhou,Z.,Liu,L.,Chandrasekhar,V.,Zhang,J.&Yi,Y.Deepreser- diesforweatherandclimatemodelling.Philos.Trans.R.Soc.A
voircomputingmeets5gmimo-ofdmsystemsinsymbol 379,20200093(2021).
NatureCommunications|(2024)15:2056 17

Perspective https://doi.org/10.1038/s41467-024-45187-1
158. Min,Q.,Lu,Y.,Liu,Z.,Su,C.&Wang,B.Machinelearning Additionalinformation
baseddigitaltwinframeworkforproductionoptimization SupplementaryinformationTheonlineversioncontains
inpetrochemicalindustry.Int.J.Inf.Manag.49, supplementarymaterialavailableat
502–519(2019). https://doi.org/10.1038/s41467-024-45187-1.
159. Kamble,S.S.etal.Digitaltwinforsustainablemanufacturing
supplychains:Currenttrends,futureperspectives,andan Correspondenceandrequestsformaterialsshouldbeaddressedto
implementationframework.Technol.Forecast.Soc.Change176, CanHuangorJieSun.
121448(2022).
160. Röhm,A.etal.Reconstructingseenandunseenattractors PeerreviewinformationNatureCommunicationsthanksSylvainGigan,
fromdataviaautonomous-modereservoircomputing.InAI andtheother,anonymous,reviewer(s)fortheircontributiontothepeer
andOpticalDataSciencesIVPagePC124380E(SPIE,Bel- reviewofthiswork.Apeerreviewfileisavailable.
lingham,2023).
161. Kong,L.W.,Weng,Y.,Glaz,B.,Haile,M.&Lai,Y.C.Reservoir Reprintsandpermissionsinformationisavailableat
computingasdigitaltwinsfornonlineardynamicalsystems. http://www.nature.com/reprints
Chaos33,033111(2023).
Publisher’snoteSpringerNatureremainsneutralwithregardtojur-
Acknowledgements isdictionalclaimsinpublishedmapsandinstitutionalaffiliations.
W.L.issupportedbytheNationalNaturalScienceFoundationofChina
(No.11925103)andbytheSTCSM(Nos.22JC1402500,22JC1401402, OpenAccessThisarticleislicensedunderaCreativeCommons
and2021SHZDZX0103).P.B.issupportedbytheEUH2020program Attribution4.0InternationalLicense,whichpermitsuse,sharing,
undergrantagreements871330(NEoteRIC),101017237(PHOENICS), adaptation,distributionandreproductioninanymediumorformat,as
101098717(Respite),101046329(NEHO),101070238(Neuropuls), longasyougiveappropriatecredittotheoriginalauthor(s)andthe
101070195(Prometheus);theFlemishFWOprojectG006020Nandthe source,providealinktotheCreativeCommonslicence,andindicateif
BelgianEOSprojectG0H1422N. changesweremade.Theimagesorotherthirdpartymaterialinthis
articleareincludedinthearticle’sCreativeCommonslicence,unless
Authorcontributions indicatedotherwiseinacreditlinetothematerial.Ifmaterialisnot
J.S.,C.H.andM.Y.initiatedthepaperanddevelopeditsoutline.J.S.,C.H. includedinthearticle’sCreativeCommonslicenceandyourintended
andM.Y.wrotethefirstdraft.P.B.,P.T.andW.L.contributedsubstantially useisnotpermittedbystatutoryregulationorexceedsthepermitted
duringthepreparationofthemanuscript.Allauthorsapprovedthe use,youwillneedtoobtainpermissiondirectlyfromthecopyright
submission. holder.Toviewacopyofthislicence,visithttp://creativecommons.org/
licenses/by/4.0/.
Competinginterests
Theauthorsdeclarenocompetinginterests. ©TheAuthor(s)2024,correctedpublication2024
NatureCommunications|(2024)15:2056 18
