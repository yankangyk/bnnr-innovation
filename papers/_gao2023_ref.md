Bioinformatics,2023,39(6),btad357
https://doi.org/10.1093/bioinformatics/btad357
Advanceaccesspublication1June2023
OriginalPaper
| Data                | and | text mining    |                                        |             |     |            |                |     |     |     |          |     |     |
| ------------------- | --- | -------------- | -------------------------------------- | ----------- | --- | ---------- | -------------- | --- | --- | --- | -------- | --- | --- |
| Similarity          |     | measures-based |                                        |             |     | graph      | co-contrastive |     |     |     | learning |     |     |
| for drug–disease    |     |                |                                        | association |     | prediction |                |     |     |     |          |     |     |
| ZihaoGao1,HuifangMa |     |                | 1,2,*,XiaohuiZhang1,YikeWang1,ZheyuWu1 |             |     |            |                |     |     |     |          |     |     |
Downloaded from https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103 by guest on 09 March 2026
1CollegeofComputerScienceandEngineering,NorthwestNormalUniversity,No.967AnningEastRoad,Lanzhou,730070,China
2GuangxiKeyLaboratoryofTrustedSoftware,GuilinUniversityofElectronicTechnology,No.1JinjiRoad,Guilin,541004,China
*Correspondingauthor.CollegeofComputerScienceandEngineering,NorthwestNormalUniversity,No.967AnningEastRoad,Lanzhou730070,China.
E-mail:mahuifang@yeah.net
AssociateEditor:ZhiyongLu
Abstract
Motivation:Animperativestepindrugdiscoveryisthepredictionofdrug–diseaseassociations(DDAs),whichtriestouncoverpotentialthera-
peuticpossibilitiesforalreadyvalidateddrugs.Itiscostlyandtime-consumingtopredictDDAsusingwetexperiments.GraphNeuralNetworks
asanemergingtechniquehaveshownsuperiorcapacityofdealingwithDDAprediction.However,existingGraphNeuralNetworks-basedDDA
predictionmethodssufferfromsparsesupervisedsignals.Asgraphcontrastivelearninghasshinedinmitigatingsparsesupervisedsignals,we
seekto leverage graphcontrastivelearningtoenhancetheprediction ofDDAs.Unfortunately, mostconventionalgraphcontrastivelearning-
basedmodelscorrupttherawdatagraphtoaugmentdata,whichareunsuitableforDDAprediction.Meanwhile,thesemethodscouldnotmodel
theinteractionsbetweennodeseffectively,therebyreducingtheaccuracyofassociationpredictions.
Results:A modelisproposedto tap potentialdrugcandidatesfor diseases, whichis calledSimilarityMeasures-basedGraphCo-contrastive
Learning(SMGCL).Forlearningembeddingsfromcomplicatednetworktopologies,SMGCLincludesthreeessentialprocesses:(i)constructs
threeviewsbasedonsimilaritiesbetweendrugsanddiseasesandDDAinformation;(ii)twographencodersareperformedoverthethreeviews,
soastomodelbothlocalandglobaltopologiessimultaneously;and(iii)agraphco-contrastivelearningmethodisintroduced,whichco-trainsthe
representationsofnodestomaximizetheagreementbetweenthem,thusgeneratinghigh-qualitypredictionresults.Contrastivelearningserves
as an auxiliary task for improving DDA predictions. Evaluated by cross-validations, SMGCL achieves pleasing comprehensive performances.
FurtherproofoftheSMGCL’spracticalityisprovidedbycasestudyofAlzheimer’sdisease.
Availabilityandimplementation:https://github.com/Jcmorz/SMGCL.
1Introduction encodes known DDAs together with drug and disease neigh-
borhoodandneighborinteractions,allowingspecificnetwork
| Rapid advances | in  | drug research | and | development | over | the |     |     |     |     |     |     |     |
| -------------- | --- | ------------- | --- | ----------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
featurestobetakenintoaccountaswell;MVGCN(Fuetal.
| past few | decades, | as well as | public | health emergencies, | such |       |            |     |          |       |              |     |                 |
| -------- | -------- | ---------- | ------ | ------------------- | ---- | ----- | ---------- | --- | -------- | ----- | ------------ | --- | --------------- |
|          |          |            |        |                     |      | 2022) | constructs |     | multiple | views | by combining |     | different simi- |
astheoutbreakofCOVID-19,haveforcedresearcherstoex-
|                   |                 |               |              |                       |             | larity    | networks        |              | with the    | biomedical | bipartite           |       | network and  |
| ----------------- | --------------- | ------------- | ------------ | --------------------- | ----------- | --------- | --------------- | ------------ | ----------- | ---------- | ------------------- | ----- | ------------ |
| plore effective   | ways            | to counter    | these        | risks. Computer-aided |             |           |                 |              |             |            |                     |       |              |
|                   |                 |               |              |                       |             | uses      | a neighborhood  |              | information |            | aggregation         | layer | to aggre-    |
| prediction        | of drug–disease |               | associations | (DDAs,                | a.k.a. drug |           |                 |              |             |            |                     |       |              |
|                   |                 |               |              |                       |             | gate      | the information |              | of          | inter-     | and intra-domain    |       | neighbors in |
| repositioning)    | is              | becoming more | appealing    | as it                 | involves    | de-       |                 |              |             |            |                     |       |              |
|                   |                 |               |              |                       |             | different | views.          |              | Although    | the        | above methods       | have  | achieved     |
| risked compounds, |                 | which could   | lead         | to lower total        | develop-    |           |                 |              |             |            |                     |       |              |
|                   |                 |               |              |                       |             | promising |                 | performance, |             | they       | all suffer sparsely |       | labeled data |
mentexpensesandshorterdevelopmentschedules.
|             |     |             |            |         |     | problems |     | due to | the limited |     | annotated | data as | wet experi- |
| ----------- | --- | ----------- | ---------- | ------- | --- | -------- | --- | ------ | ----------- | --- | --------- | ------- | ----------- |
| At present, | the | popular DDA | prediction | methods | can | be       |     |        |             |     |           |         |             |
mentsareexpensiveandtime-wasting.Thesedataareinsuffi-
roughly divided into two categories: DDA prediction based cienttoinduceaccuraterepresentationsofdrugsanddiseases
| on matrix | decomposition | and | completion, | and | DDA predic- |     |     |     |     |     |     |     |     |
| --------- | ------------- | --- | ----------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
inmostcases,leadingtosuboptimalperformance.
tionbasedonGraphNeuralNetworks(GNNs).Forthemeth-
|           |           |               |     |                 |      |        | A contrastive |              | learning | paradigm      | from  | the computer | vision |
| --------- | --------- | ------------- | --- | --------------- | ---- | ------ | ------------- | ------------ | -------- | ------------- | ----- | ------------ | ------ |
| ods based | on matrix | decomposition |     | and completion, | BNNR |        |               |              |          |               |       |              |        |
|           |           |               |     |                 |      | domain | is            | one approach |          | to addressing | these | difficulties | (Wu    |
(Yang et al. 2019) integrates the drug–drug, drug–disease, etal.2018,Chenetal.2020),whichaimstoconstructconsis-
and disease–disease networks and uses a bounded nuclear tent and inconsistent view pairs via data augmentations, in-
normregularizationmethodtocompletethedrug–diseasema-
|     |     |     |     |     |     | cluding | cutout | and | color | distortion | (Howard |     | 2014). Some |
| --- | --- | --- | --- | --- | --- | ------- | ------ | --- | ----- | ---------- | ------- | --- | ----------- |
trix under the low-rank assumption; GRGMF (Zhang et al. researchers have made a preliminary attempt at graph data
2020b) is an improved neural collaborative filtering frame- (Huang et al. 2021, Zhao et al. 2021). However, contrastive
work, which learns the neighbor information for each node learning on drug repositioning has its unique challenges: (i)
adaptively and draws support from existing external similar- thegraphofDDAshasfewernodesandmoresparseedges(a
ity information to enhance the prediction performance. For number of diseases might only be treated by one drug).
themethodsbasedonGNNs,DRWBNCF(Mengetal.2022) Therefore,techniqueswithnode/edgedropoutarecompletely
Received:September13,2022.Revised:March14,2023.EditorialDecision:May20,2023.Accepted:May31,2023
VC TheAuthor(s)2023.PublishedbyOxfordUniversityPress.
ThisisanOpenAccessarticledistributedunderthetermsoftheCreativeCommonsAttributionLicense(https://creativecommons.org/licenses/by/4.0/),which
permitsunrestrictedreuse,distribution,andreproductioninanymedium,providedtheoriginalworkisproperlycited.

2 Gaoetal.
unavailable for DDA prediction. (ii) When creating self- similar to this drug in chemical structure, side effects, etc. In
supervisionsignals,mostexistingmethodsgenerallyconsider thisway,thedrug-similarityviewisdenotedasGR 2fVR;ERg
neighborsinisolation.Weinsteadarguethatinteractionsbe- with N drugs, and its adjacency matrix AR 2f0;1gN(cid:4)N,
tween neighboring nodes may reveal potential relations be- whereAR ¼1ifdrugr isthetop-Knearestneighborofdrug
ij j
tween them and the target node, and modeling such r; otherwise AR ¼0. In the same way, the disease-similarity
i ij
interactionscanimprovethetargetnoderepresentationtoim- viewisdenotedasGD 2fVD;EDgwithMdiseases,anditsad-
plyrichersemantics. jacencymatrixAD 2f0;1gM(cid:4)M,whereAD ¼1ifdiseased is
ij j
To get over the aforementioned limitations, we enrich the the top-K nearest neighbor of disease d; otherwise AD ¼0.
i ij
DDA graph contrastive learning (GCL) by incorporating the Fordescriptivepurposes,wedefinetermsthatareusedinter-
drug–drug similarity graph and disease–disease similarity changeably throughout the literature: view is a synonym for
graph, motivated by the fact that the indications for similar graph.
drugsareoftenthesame.Ontopofthat,weproposeanend-
to-end Similarity Measures-based Graph Co-contrastive 2.2Context-awareneighborhoodaggregation
Learning (SMGCL) model for DDA prediction with three After views construction, we develop a context-aware neigh-
modules.Thefirstmodule,“multi-sourcecontrastviewscon- borhoodaggregationincludingtwoencoders,tocaptureboth
struction,” builds the known DDA view, the drug-similarity, heterogeneous(homogeneous)andlocal(global)information.
and disease-similarity views (applying the nearest neighbors) Each encoder is in charge of extracting useful information
byusingthreesourcesofdata.Thesecondmodule,“context- from one heterogeneous (homogeneous) graph to improve
aware neighborhood aggregation,” uses a bilinear GNN to DDAprediction.
capture complicated local feature in the DDA view, and a
global-aware attention mechanism to compensate for the re- 2.2.1Nodefeatureextraction
ceptive field issue in bilinear aggregation. The last module is Each column of the adjacency matrix of the similarity view
“contrastiveobjective,”whereweintroduceasamplingmech- canactasaninitialfeaturevectorforthecorrespondingnode;
anism to radically mine supervised signals for efficient co- however,thesevectorsmaynotcapturethehigherordercon-
contrastivelearning.Furthermore,thepredictiontaskandthe nectivity information of the graph. For this reason, we run
contrastive learning task are unified under a Random Walk with Restart (Tong et al. 2006) separately on
“primary&auxiliary” learning paradigm. Cross-validation drug-similaritymatrixAR anddisease-similaritymatrixAD to
and extensive experiments on three benchmark datasets pro- enrich the initial embeddings for each node with local struc-
vide statistical evidence for the superiority of SMGCL over
ture context. The process can be defined as the following re-
thebaselineapproaches,andfurthercasestudydemonstrates
currenceequation:
thepracticabilityofSMGCL.
xðlþ1Þ ¼ð1(cid:5)aÞPRxðlÞþa(cid:6)xð0Þ; (1)
ri ri ri
2Materialsandmethods
whereaistherestartprobability,PR istheprobabilitytransi-
Wedenotevectorsbylowercaseboldface,matricesbyupper-
tionmatrixobtainedfromAR bycolumn-wisenormalization.
case boldface, and sets by uppercase calligraphic font. Thus,
l t e h t e R nu ¼ mb fr e 1 r ; o r 2 f ; d . r . u .; g r s N ; g D d ¼ en f o d t 1 e ; s d t 2 h ; e .. s . e ; t d o M f g d d r e u n g o s, te w s h th e e re se N t o is f x ca ð ri l t Þ es is th a e c p o r lu o m ba n bi v li e t c y to o r fr o e f a d ch ru in g g n n o o d d e e r i i , a w fte h r o l se ste it p h s. e x n r ð t 0 i r Þ y 2 in R d N i-
diseases,whereMisthenumberofdiseases.Theobjectiveof isaone-hotvectorwithdimensionsofNwhereithentryis1
DDA prediction is to learn a mapping function fððr;dÞjxÞ: and 0 otherwise, which denotes the initial vector representa-
E !½0;1(cid:2)fromedgestoscores,wherexisaparameter,inor- tionofdrugr i .
der to determine the probability that a given drug would be After approaching the steady-state, a single-layer percep-
e
c
f
h
f
i
e
t
c
e
t
c
i
t
v
u
e
re
in
of
tr
t
e
h
a
e
ti
p
n
r
g
o
a
po
g
s
i
e
v
d
en
m
d
e
i
t
s
h
e
o
a
d
se
.
.
N
F
o
ig
te
ur
t
e
ha
1
t,
d
t
i
h
s
e
pl
d
a
e
y
s
s
cr
th
ip
e
ti
a
o
r
n
- t
w
ro
h
n
ere
is
e r
a
i
p
2
pl
R
ie
t
d
de
to
no
o
te
b
s
ta
t
i
h
n
e
e
u
r
p
i
d
¼
at
M
ed
L
d
P
r
ð
u
x
g
1 ri Þ
no
o
d
n
e
A
re
R
pr
f
e
o
s
r
en
d
ta
ru
ti
g
o
s
n
,
on the whole model from the drug part, since the drug and with t dimensions and MLP contains single hidden layer. In
diseasepartsaredual. thesame way,we can obtainthe disease noderepresentation
e 2Rt.
dj
2.1Multi-sourcecontrastviewsconstruction
2.1.1DDAview 2.2.2DDAviewencoder
The DDA view can be regarded as an undirected graph GCN (Kipf and Welling 2017) assumes that neighboring
G¼fV;Eg, where V represents the set of nodes that corre- nodesareindependentofeachotherandutilizestheweighted
spond to drugs and diseases, E (cid:3)V(cid:4)V denotes the set of sum to learn low-dimensional representations of nodes. We
edges and indicates the existence of interaction between two formulateaGAaggregatorfortargetnodev(drugrordisease
kinds of nodes in V. Furthermore, the graph G can be repre- d)as:
sentedasanincidencematrixA2f0;1gN(cid:4)M,whereA ¼1if
ij
drugr i cantreatdiseased j ,otherwiseA ij ¼0. hðGAÞ ¼GAðfeg Þ¼r X a W e ! ; (2)
v i i2N^ðvÞ vi g i
2.1.2Similarityview
i2N^ðvÞ
Atremendousdealofefforthasgoneintocalculatingthesimi-
^
larity of drugs or diseases. Taking the construction of drug- where GAð(cid:6)Þ is the non-linear aggregator, NðvÞ¼
similarityviewasanexample,withthesimilarityofdrugs,for fvg[fijA ¼1g denotes the extended neighbors of node v,
vi
a certain drug node r, we can select drugs with the top-K which contains the node v itself. r is a non-linear activation
i
highest similarity as the neighbor nodes, which are the most function. a is the weight of neighbor i and is defined as
vi
Downloaded
from
https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103
by
guest
on
09
March
2026

| SMGCL |     |     |     |     |     |     |     |     |     |     |     |     |     | 3   |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Downloaded from https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103 by guest on 09 March 2026
Figure1.TheframeworkoverviewoftheproposedSMGCL.Solidroundedrectanglesin(a)indicatethreekindsofviews,whichareconstructedfrom
threedifferentkindsofdata.TheDDAviewisconstructedontheknownassociationsinthetrainingset.Next,thenoderepresentationgeneratedbythe
randomwalkwithrestartistransformedandappliedasinputtothemodel.Then,filledroundedrectanglesin(b)indicateneuralnetworkencoders.For
eachtypeofnode,wecangettwokindsofrepresentationsbythedifferentneuralnetworkencoders.Finally,weco-trainthenoderepresentations,the
predictiontaskandthecontrastivelearningtaskareunifiedunderaprimary&auxiliarylearningparadigmin(c).Bestviewedincolor.
^ ^ ^ ^ d i s s i m i la rd r u g s m ig h t a l s o tr e a t th e sam e d i sea se . T o f u l l y ex -
| p ffi 1 ffiffi ffi ffiffiffi , | where d | ¼jN ðvÞj | and | d ¼jN | ðiÞj. W | is the | weight |     |     |     |     |     |     |     |
| ------------------------------ | ------- | -------- | --- | ----- | ------- | ------ | ------ | --- | --- | --- | --- | --- | --- | --- |
| d^ d^                          | v       |          |     | i     |         | g      |        |     |     |     |     |     |     |     |
v i p l o i t t h is p o t e n tia l c o r r e la ti o n , w e de si g n a g l o ba l - a w a re
matrixtodofeaturetransformation.
|              |     |               |     |        |             |     |       | strategy | based | on an attention |     | architecture, | which increases |     |
| ------------ | --- | ------------- | --- | ------ | ----------- | --- | ----- | -------- | ----- | --------------- | --- | ------------- | --------------- | --- |
| In addition, | the | co-occurrence |     | of two | neighboring |     | nodes |          |       |                 |     |               |                 |     |
significantsignalsandweakensnoisysignalswhencalculating
can be regarded as an important feature of the target node. the attention coefficient d , to obtain node representations
vi
However,thecommonGCNsignorethepossibleinteractions
|          |             |                |      |           |            |             |     | considering |         | various perspectives. |      | Specifically, | the following    |     |
| -------- | ----------- | -------------- | ---- | --------- | ---------- | ----------- | --- | ----------- | ------- | --------------------- | ---- | ------------- | ---------------- | --- |
| between  | neighboring | nodes.         | Even | if it     | is a Graph | Attention   |     |             |         |                       |      |               |                  |     |
|          |             |                |      |           |            |             |     | two         | aspects | are taken             | into | account       | by the attention |     |
| Networks | that        | can adaptively |      | aggregate | the        | information | of  |             |         |                       |      |               |                  |     |
mechanism.
| neighboring | nodes | of  | different | importance, | it  | cannot | extract |     |     |     |     |     |     |     |
| ----------- | ----- | --- | --------- | ----------- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- |
Firstly,wecalculatetheaveragerepresentationofallnodes’
| the possible | interaction |             | features | between | neighboring |                 | nodes. |             |     |                |       |          |            |         |
| ------------ | ----------- | ----------- | -------- | ------- | ----------- | --------------- | ------ | ----------- | --- | -------------- | ----- | -------- | ---------- | ------- |
|              |             |             |          |         |             |                 |        | embeddingin |     | the similarity | view. | In order | to explore | the po- |
| At the       | same time,  | multiplying |          | two     | vectors     | can effectively |        |             |     |                |       |          |            |         |
tentialofdrugtreatmentfornon-indications,thenoderepre-
modeltheinteractionsbyemphasizingtheconsistentinforma-
|          |           |     |           |             |     |      |        | sentationandaverageinformation |     |     |     | representationareusedto |     |     |
| -------- | --------- | --- | --------- | ----------- | --- | ---- | ------ | ------------------------------ | --- | --- | --- | ----------------------- | --- | --- |
| tion and | weakening | the | divergent | information |     | (Zhu | et al. |                                |     |     |     |                         |     |     |
calculatethefollowingattentionscore:
2020).Thus,wedefineaBAaggregatorfortargetnodevas:
|       |     |               |         |     |     |     |     |     |     | (cid:2) ¼att | ðW  | e (cid:7)eÞ; |     | (5) |
| ----- | --- | ------------- | ------- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ | --- | --- |
| hðBAÞ |     | (cid:3)       | (cid:4) |     |     |     |     |     |     | i            | 1   | 1 i          |     |     |
|       | ¼BA | fhg i i2N^ðvÞ |         |     |     |     |     |     |     |              |     |              |     |     |
v
 1 ! where att is a single-layer feedforward neural network with
|     |     | X   | X   |     |            |     | (3) |     | 1   |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | ¼r  |     |     | W   | e (cid:7)W | e ; |     |     |     |     |     |     |     |     |
b b i b j theLeakyReLUasactivationfunction,W 1 isatransformation
v
i2N^ðvÞj2N^ðvÞ&i<j matrix,e representstheaveragenodeinformationbyaverage
pooling.
where BAð(cid:6)Þ is the non-linear aggregator, b ¼1d ^ ðd ^ (cid:5)1Þ Apartfromtheabove,weextendthemessagepassingpro-
|     |     |     |     |     |     | v 2 v | v   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
denotesthenumberofinteractionsforthenodev,eliminating cessbytheattentionmechanism.Ifthedrugneighbornodeis
thebiasofnodedegreetosomeextentwiththenormalization morecorrelatedwiththetargetdrugnode,itscontributionin
process.(cid:7)iselement-wiseproductandW istheweightma- aggregation toward the target node will be more significant
b
| trixtodofeaturetransformation. |     |     |     |     |     |     |     | andviceversa. |     |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
Then,theencoderwhichisbuiltontheDDAviewformes-
| sage passing | between |     | drugs | and diseases |     | extracts | indirect |     |     |        |      |            |       |     |
| ------------ | ------- | --- | ----- | ------------ | --- | -------- | -------- | --- | --- | ------ | ---- | ---------- | ----- | --- |
|              |         |     |       |              |     |          |          |     |     | f ¼att | 2 ðW | 2 ejjW i 2 | eÞ; j | (6) |
ij
interactionsinthelocalstructure.Specifically,fortargetnode
v,theDDAviewencoderisdefinedas: where W is a transformation matrix, k denotes the concate-
2
|     |     |     |     |     |     |     |     | nationoperation,e |     | j istheneighbornoderepresentationofthe |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | -------------------------------------- | --- | --- | --- | --- |
¼b(cid:4)hðGAÞþð1(cid:5)bÞ(cid:4)hðBAÞ;
|     | h   |     |     |     |     |     | (4) | nodev,andatt |     | isasingle-layerfeedforwardneuralnetwork |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --------------------------------------- | --- | --- | --- | --- |
|     | v   |     | v   |     | v   |     |     |              |     | 2                                       |     |     |     |     |
applyingtheLeakyReLUnonlinearity.
wherebisahyper-parametertotrade-offthestrengthsofthe Then,boththeglobalandlocalscoreofeachnodeisadded
GAaggregatorandBAaggregator. following the additive attention mechanism (Bahdanau et al.
2015).Besides,softmaxfunctionisutilizedtonormalizecoef-
2.2.3Similarityviewencoder ficients across all choices of j, so as to make coefficients are
Previous drug repositioning research assumed that similar able to directly compared between all nodes. The attention
drugs would treat the same disease, but we argue that coefficientsd betweennodeiandnodejcanbecalculatedas:
ij

4 Gaoetal.
d
ij
¼softmax
i
ð(cid:2)
j
þf
ij
Þ¼P exp
e
ð
x
(cid:2) j
p
þ
ð(cid:2)
f ij
þ
Þ
f Þ
; (7) m
co
i
n
n
s
im
id
i
e
z
r
in
th
g
e
t
v
h
a
e
ri
w
an
e
t
ig
o
h
f
te
o
d
ur
b
m
in
o
a
d
ry
el,
c
n
ro
a
s
m
s-
e
e
d
nt
S
r
M
op
G
y
C
lo
L
s
-
s
N
,
S
w
,
e
wh
a
i
l
c
so
h
k2NðiÞ k ik
minimizesthebinarycross-entropyloss.Italsomeansthatthe
where (cid:2) determines the amount of information flow from j same number of unknown DDAs as known associations is
j
while f decides the information target node i may receive. In sampled.
ij
this way, we can get another representation of drugs and dis-
2.4Contrastiveobjective
easesobtainedon“drug-similarityview”and“disease-similarity
2.4.1Miningself-supervisionsignals
view,” respectively, which are denoted as q ðv2fr;dgÞ. The
v Through the above section, we have constructed two view
calculationisdefinedas:
encoders over three views, each of which can deliver compli-
(cid:5)X (cid:6) mentarysemanticstotheother.Asaresult,itmakessenseto
q ¼r d W h ; (8)
i ij 3 j improveeachencoderbyusingthedatafromtheotherview.
j2NðiÞ
Inthissection,weillustratehowSMGCLenhancesDDApre-
diction by mining valuable self-supervision signals. This can
whereW istheweightmatrix.
3 be accomplished by following the co-training architecture.
Given a drug r and disease d in the DDA view, we choose
REMARK:Weelaboratelydescribethedrugrepresentation i j
theirpositiveandnegativedrugsampleswithinthesamemini-
learningprocesshere.Becausethedisease
batchusingitsrepresentationlearnedoverthesimilarityview:
representationlearningisadualprocess,weomititfor
brevity.
score ¼softmaxðQ q Þ; (11)
r d r
2.3Generatingpredictionandmodeloptimization where score r 2RM denotes the predicted probability of each
diseasebeingcuredtothedrugrinthesimilarityview.
To reconstruct the associations between drugs and diseases,
A natural intuition is that we may select highly confident
ourdecoderfðe ;e Þisformulatedasfollows:
ri dj
diseases via calculated probabilities, i.e. top-K ranking dis-
eases,soastosupervisethedrugembeddinginthesimilarity
^y ¼MLPðe (cid:7)e ;h ;h Þ; (9)
ri;dj ri dj ri dj view as augmented ground truths. The positive sample selec-
tionisdefinedas:
where^y isthepredictedprobabilityscore.
ri;dj
DDA graph possesses two characteristics: (i) sparse edges
Sdþ ¼PKðscore Þ; (12)
(there is only a small number of existing DDAs) and (ii) lim- ri d ri
itednodes(thenumberofdrugsanddiseasesarefarlessthan
those of users and items in the recommendersystems). In or- wherePK d denotespickingthecorrespondingdiseasesd,which
dertomakefulluseofalltheseinformation,wethustakeall areaccordingtothetop-Kprobabilityscoreswiththehighest
unknowndrug–diseasepairsasnegativeinstancesinthetrain- confidence.
ing set of each fold. Since there is no negative sampling, the Whenitcomestopickingnegativesamples,asimpleintui-
setting of negative samples inthe training set and the test set tion is to choose the diseases with the lowest scores.
arethesame.Furthermore,someoftheexistingstudies(Zhao Nevertheless,thisapproachcontributesminimallytotherep-
et al. 2021) sample the same number of unknown DDAs as resentationupdateandcannotdistinguishandtailorcomplex
thatoftheknownassociationinthetrainingsetinsomestud- anddifficultsamples.Thus,Knegativesamplesarerandomly
i
d
e
o
s.
m
W
sa
e
m
a
p
rg
li
u
n
e
g,
th
w
a
h
t
ic
th
h
e
is
sa
li
m
ke
p
ly
lin
to
g
i
s
n
tr
tr
a
o
te
d
g
u
y
ce
te
u
n
n
d
n
s
e
t
c
o
es
a
sa
d
r
o
y
p
n
t
o
ra
is
n
e
-
.
c
th
h
e
os
p
e
o
n
si
f
t
r
i
o
v
m
es
d
to
ise
c
a
o
s
n
es
st
r
r
a
u
n
ct
ke
S
d
d r (cid:5)
in
. W
to
e
p
a
5
r
0
g
%
ue
in
th
s
a
c
t
o
t
r
h
e
e
ri
se
ex
d
c
i
l
s
u
e
d
a
i
s
n
e
g
s
GiventhattherearefarfewerknownDDAsthanthereareun- shouldbeconsideredashardnegatives,suggestingfinerinfor-
knownorunseenDDAs,andsinceknownDDAshaveunder- mation with slight possibility of false negatives that may de-
gone extensive laboratory and clinical validation, they are ceive learning. Finally, the information samples used for
highly reliable and crucial for enhancing predictive perfor- disease embeddings are selected in the same way to get Srþ
mance. Hence, our proposed SMGCL learns parameters by
andSr(cid:5). di
di
minimizingtheweightedbinarycross-entropylossasfollows: The positive and negative pseudo-labels for each drug and
diseaseinthesimilarityviewarerepeatedlygeneratedforev-
1 X X ! ery training batch. More hard negative samples are antici-
L ¼(cid:5) g(cid:4) log^y þ ð1(cid:5)log^y Þ ;
bce N(cid:4)M ri;dj ri;dj pated to be produced by repeating this procedure. Note that
ði;jÞ2Sþ
rd
ði;jÞ2S(cid:5)
rd the encoders can evolve under the guidance of informative
(10) samples,recursivelyextractingmorehardsamples.
where ði;jÞ indicates the pair of drug r and disease d, Sþ 2.4.2Co-contrastivelearning
i j rd
denotesthesetofallknownDDAs,andS(cid:5) representstheset With the generated pseudo-labels, the graph co-contrastive
rd
of all unknown or unseen DDAs. The balance factor g¼ jS(cid:5) rd j learning task for evolving the encoder can be performed by
jSþj contrastive objects. We utilize NT-Xent (You et al. 2020) as
rd
emphasizes the importance of observed associations to miti- our objective function to maximize the mutual information
gate the damageof data imbalance,wherejS(cid:5) rd j andjSþ rd j are between the two views. Formally, the training objective for
the number of pairs in S(cid:5) and Sþ. Moreover, instead of drugh isasfollows:
rd rd ri
Downloaded
from
https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103
by
guest
on
09
March
2026

| SMGCL |     |     |     |         |     |         |     |     |     |     |     |     |     |     | 5   |     |
| ----- | --- | --- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|       |     |     |     | (cid:7) |     | (cid:8) |     |     |     |     |     |     |     |     |     |     |
P e si m ð ð hr i ; h d j ÞÞ = s a n d th e A r e a U n d e r t h e P r e c i s i o n – R e c a l l c u r v e ( A U P R ) a s p r i -
|     |     |     | d 2 | S d þ |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
L ¼(cid:5)log j ri ; (13) m a r y m e t r i c s . I t i s m e a n i n g f u l t o m e a s u r e t h e c h a r a c t e r i s t i c s
|     | ri  |     | P   |     | (cid:7) | (cid:8) |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
e s im ð h r i ; h dk Þ =s o f R O C a n d P R w h i l e t r e a t i n g t h e u n k n o w n s a s t r u e n e g a t i v e s
|     |     |     | dk 2 S | d þ [Sd (cid:5) |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
r i ri si n c e t h e a c t u a l a s s o c ia t io n s a r e l i m i t e d i n c o m p a ri s o n t o t h e
s simðu;vÞ total number of unknowns. Details of each metric are in the
| where | denotes | the temperature |     | parameter |     | and | is  |     |     |     |     |     |     |     |     |     |
| ----- | ------- | --------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
SupplementaryMaterial.
| the cosine  | similarity.  | In  | the same | way, | the | training | objective |     |          |       |       |      |     |      |            |     |
| ----------- | ------------ | --- | -------- | ---- | --- | -------- | --------- | --- | -------- | ----- | ----- | ---- | --- | ---- | ---------- | --- |
|             |              |     |          |      |     |          |           | Our | proposed | SMGCL | model | uses | the | Adam | optimizer. |     |
| fordiseaseh | isdefinedas: |     |          |      |     |          |           |     |          |       |       |      |     |      |            |     |
di The values of all hyper-parameters refer to the practices of
(cid:7) (cid:8) previous researchers and are finally determined by grid
|     |     |     | P   | e si | m ð ð hd ;h rj ÞÞ | = s |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
r j2 S r þ i s e a r c h , w h e r e t h e l e a r n i n g r a t e i s s et a s 0 . 0 0 1 , b a t c h s iz e is s e t
|     |     | ¼(cid:5)log |     | d i |     | :   | (14) |     |     |     |     |     |     |     |     |     |
| --- | --- | ----------- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
L di P a s 6 4 , r es t a r t p r o b a b il i ty a ¼ .1 , t e m p e r a t u r e s ¼ 0 :1 , a n d
|     |     |     |     | ð   | e s im ð h di ;h | rk Þ =sÞ |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
rk 2 S r þ [Sr (cid:5) s c a l e c o n tr o l h y p e r -p a r a m e te r k ¼ 0 : 1. F o r t r a d e - o f f h y p e r -
|     |     |     |     | d i d i |     |     |     |           |          |     |               |     |         |            |     |                                                                                                                |
| --- | --- | --- | --- | ------- | --- | --- | --- | --------- | -------- | --- | ------------- | --- | ------- | ---------- | --- | -------------------------------------------------------------------------------------------------------------- |
|     |     |     |     |         |     |     |     | parameter | b, SMGCL |     | has different |     | optimal | parameters | for | Downloaded from https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103 by guest on 09 March 2026 |
Finally,weunifythepredictiontaskwiththeauxiliarySSL
|     |     |     |     |     |     |     |     | different | benchmark |     | datasets. | On  | Fdataset, | b¼0:6; | on  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --- | --------- | --- | --------- | ------ | --- | --- |
task.ThetotallossLisdefinedas: CdatasetandLRSSL,b¼0:8.Besides,allmethodshavebeen
comparedunderthesameevaluationsettings.Forthebaseline
|     |     | L¼L | þk(cid:6)ðL |     | þL Þ; |     | (15) |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ----------- | --- | ----- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bce r d models available for code disclosure, we run the code with
|     |     |     |     |     |     |     |     | reference | to the | best parameters |     | reported |     | in the | original pa- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | --------------- | --- | -------- | --- | ------ | ------------ | --- |
where k is hyper-parameter to control the scale of the graph per,andourresultsareconsistentwiththoseinpublications.
co-training. Forthebaselinemodelswithunavailablecodes,wereportthe
| The weights | are | initialized |     | in accordance |     | with Glorot | and |     |     |     |     |     |     |     |     |     |
| ----------- | --- | ----------- | --- | ------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
resultsdirectlysinceweusethesamedatasets.
| Bengio (2010), |     | and the | model | is optimized |     | using the | Adam |     |     |     |     |     |     |     |     |     |
| -------------- | --- | ------- | ----- | ------------ | --- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
optimizer (Kingma and Ba 2015). We train the model in a 3.2Overallperformance
denoisingsetupbyrandomlydroppingoutedgeswithafixed FollowingLietal.(2020)andZhangetal.(2020a),weadopt
probability, which enables us to effectively generalize to the 10-foldcross-validation(10-CV)toevaluatetheperformance
| unseen data | and | avoid | the model | from | over-fitting. |     | For the |               |          |     |                |     |     |            |         |     |
| ----------- | --- | ----- | --------- | ---- | ------------- | --- | ------- | ------------- | -------- | --- | -------------- | --- | --- | ---------- | ------- | --- |
|             |     |       |           |      |               |     |         | of prediction | methods. |     | In particular, |     | for | each 10-CV | repeti- |     |
graphconvolutionlayers,wealsouseregulardropout.
tion,wecalculateallevaluationmetrics,andthefinalevalua-
tionresultsareobtainedbycalculatingtheaverageevaluation
|     |     |     |     |     |     |     |     | metrics | over 10 | repetitions. |     | The | prediction | model | is con- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ------------ | --- | --- | ---------- | ----- | ------- | --- |
3Experiments
|     |     |     |     |     |     |     |     | structed | on the | known | associations |     | in the | training | set and is |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------ | ----- | ------------ | --- | ------ | -------- | ---------- | --- |
3.1Experimentalsettings
|     |     |     |     |     |     |     |     | used to | predict | the associations |     | in  | the remaining |     | fold as the |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ---------------- | --- | --- | ------------- | --- | ----------- | --- |
3.1.1Datasets
testset.Besides,wedeployat-testunderAUROCandAUPR
| We evaluate | our       | model | on  | three  | benchmark  |      | datasets: |                 |               |         |     |             |          |            |            |     |
| ----------- | --------- | ----- | --- | ------ | ---------- | ---- | --------- | --------------- | ------------- | ------- | --- | ----------- | -------- | ---------- | ---------- | --- |
|             |           |       |     |        |            |      |           | metrics.        | Table 2       | reports | the | performance |          | comparison | results    |     |
| “Fdataset”  | (Gottlieb | et    | al. | 2011), | “Cdataset” | (Luo | et al.    |                 |               |         |     |             |          |            |            |     |
|             |           |       |     |        |            |      |           | and statistical | significance, |         | in  | which       | SMGCL-NS |            | means that |     |
2016),and“LRSSL”(Liangetal.2017),whichareoftenused
thesamenumberofunknownDDAsasknownassociationsis
| in DDA | prediction. | The | basic | statistics | of the | three | datasets |     |     |     |     |     |     |     |     |     |
| ------ | ----------- | --- | ----- | ---------- | ------ | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
sampled.Wehavethefollowingobservations:
| are shown | in Table | 1.  | Sparse | ratio | is defined | as the | ratio of |     |     |     |     |     |     |     |     |     |
| --------- | -------- | --- | ------ | ----- | ---------- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thenumberofknownassociationstothenumberofallpossi-
|                   |     |         |     |       |            |     |        | 1) On | three | datasets, | BNNR | and | DRIMC |     | outperform |     |
| ----------------- | --- | ------- | --- | ----- | ---------- | --- | ------ | ----- | ----- | --------- | ---- | --- | ----- | --- | ---------- | --- |
| ble associations. |     | Details | of  | these | benchmarks | are | in the |       |       |           |      |     |       |     |            |     |
expectationsintermsofperformance.Suchperformance
SupplementaryMaterial.
|     |     |     |     |     |     |     |     | might | be   | attributed | to  | a smaller  | number |            | of nodes in |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---- | ---------- | --- | ---------- | ------ | ---------- | ----------- | --- |
|     |     |     |     |     |     |     |     | DDA   | data | compared   | to  | e-commerce |        | and social | recom-      |     |
3.1.2Baselinemethods
To evaluate the effectiveness of our proposed SMGCL, we mendation data, which allows for the promising perfor-
|         |         |         |          |          |     |        |          | mance | of BNNR |     | and DRIMC |     | on AUROC. |     | In addition, |     |
| ------- | ------- | ------- | -------- | -------- | --- | ------ | -------- | ----- | ------- | --- | --------- | --- | --------- | --- | ------------ | --- |
| compare | it with | various | baseline | methods: | (i) | matrix | factori- |       |         |     |           |     |           |     |              |     |
asanimprovedneuralcollaborativefilteringframework,
| zation and | completion |     | models | including | SCMFDD |     | (Zhang |       |            |     |     |       |                |     |          |     |
| ---------- | ---------- | --- | ------ | --------- | ------ | --- | ------ | ----- | ---------- | --- | --- | ----- | -------------- | --- | -------- | --- |
|            |            |     |        |           |        |     |        | GRGMF | introduces |     | two | graph | regularization |     | terms to |     |
etal.2018),BNNR(Yangetal.2019),DRIMC(Zhangetal.
2020a),andGRGMF(Zhangetal.2020b);(ii)deeplearning- deal with nodes without any known link information,
basedmodelsincludingNIMCGCN(Lietal.2020),LAGCN
Table2.Theaveragemetricsofcomparedmethodsobtainedin10-CV.
| (Yu et | al. 2021), | DRWBNCF    |         | (Meng | et    | al. 2022), | and     |         |     |          |     |          |     |     |       |     |
| ------ | ---------- | ---------- | ------- | ----- | ----- | ---------- | ------- | ------- | --- | -------- | --- | -------- | --- | --- | ----- | --- |
| MVGCN  | (Fu et     | al. 2022). | Details | of    | these | baseline   | methods |         |     |          |     |          |     |     |       |     |
|        |            |            |         |       |       |            |         | Dataset |     | Fdataset |     | Cdataset |     |     | LRSSL |     |
areintheSupplementaryMaterial.
|     |     |     |     |     |     |     |     |     | AUROC | AUPR |     | AUROC | AUPR | AUROC | AUPR |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---- | --- | ----- | ---- | ----- | ---- | --- |
3.1.3Evaluationmetricsandparameterssettings
|           |         |         |              |     |     |       |          | SCMFDD | 0.7748 | 0.0510 |     | 0.7921 | 0.0514 | 0.7783 | 0.0358 |     |
| --------- | ------- | ------- | ------------ | --- | --- | ----- | -------- | ------ | ------ | ------ | --- | ------ | ------ | ------ | ------ | --- |
| To assess | SMGCL’s | overall | performance, |     | we  | adopt | the Area |        |        |        |     |        |        |        |        |     |
|           |         |         |              |     |     |       |          | BNNR   | 0.9298 | 0.4372 |     | 0.9338 | 0.4702 | 0.9267 | 0.3152 |     |
UndertheReceiverOperatingCharacteristiccurve(AUROC) DRIMC 0.9091 0.3096 0.9333 0.3894 0.9314 0.2661
|     |     |     |     |     |     |     |     | GRGMF   | 0.8047 | 0.5503 |     |        |        | 0.8157 | 0.4396 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ------ | --- | ------ | ------ | ------ | ------ | --- |
|     |     |     |     |     |     |     |     | NIMCGCN | 0.8281 | 0.3385 |     | 0.8508 | 0.4326 | 0.8294 | 0.2670 |     |
Table1.Statisticaldetailsofthebenchmarkdatasets.
|     |     |     |     |     |     |     |     | LAGCN | 0.8586 | 0.1188 |     | 0.9144 | 0.1849 | 0.9336 | 0.1109 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------ | ------ | --- | ------ | ------ | ------ | ------ | --- |
Dataset Numberof Numberof Numberof Sparse MVGCN 0.8527 0.5582 0.8617 0.6302 0.8493 0.4431
drugs diseases associations ratio DRWBNCF 0.9245 0.4845 0.9404 0.5589 0.9345 0.3416
|          |     |     |     |     |      |     |        | SMGCL    | 0.9352* | 0.5486* |     | 0.9468* | 0.6256* | 0.9262* | 0.3904* |     |
| -------- | --- | --- | --- | --- | ---- | --- | ------ | -------- | ------- | ------- | --- | ------- | ------- | ------- | ------- | --- |
| Fdataset | 593 |     | 313 |     | 1933 |     | 0.0104 |          |         |         |     |         |         |         |         |     |
|          |     |     |     |     |      |     |        | SMGCL-NS | 0.9284* | 0.5244* |     | 0.9369* | 0.5816* | 0.9136* | 0.4374* |     |
| Cdataset | 663 |     | 409 |     | 2352 |     | 0.0087 |          |         |         |     |         |         |         |         |     |
LRSSL 763 681 3051 0.0058 * IndicatesP-value<.05inthesignificancetest.Thebestresultsarein
bold,andthesuboptimalresultsareunderlined.

| 6                                     |           |              |           |            |                  |          |     |     |     | Gaoetal. |
| ------------------------------------- | --------- | ------------ | --------- | ---------- | ---------------- | -------- | --- | --- | --- | -------- |
| thus enhancing                        |           | the learning |           | of latent  | representations. |          |     |     |     |          |
| This may                              | greatly   | alleviate    | the       | influence  | of unbalanced    |          |     |     |     |          |
| data on                               | the model | and          | achieve   | suboptimal | performance      |          |     |     |     |          |
| onAUPR.However,GRGMFdoesnotexplicitly |           |              |           |            |                  | model    |     |     |     |          |
| the connectivity                      |           | in the       | embedding |            | learning         | process, |     |     |     |          |
whicheasilyleadstoitspoorperformanceonAUROC.
| 2) Compared  | with       | NIMCGCN     |          | and LAGCN,     | the | perfor-   |     |     |     |     |
| ------------ | ---------- | ----------- | -------- | -------------- | --- | --------- | --- | --- | --- | --- |
| mance        | of DRWBNCF |             | verifies | that modeling  |     | neighbor  |     |     |     |     |
| interactions |            | can improve |          | representation |     | learning. |     |     |     |     |
MVGCNistheonlymodelthatusescontrastivelearning
| apart from | the | proposed | SMGCL. | The | difference | with |     |     |     |     |
| ---------- | --- | -------- | ------ | --- | ---------- | ---- | --- | --- | --- | --- |
SMGCListhatMVGCNusescontrastivelearningtoob-
| tain the | initial | representation |     | of nodes, | while | SMGCL |     |     |     |     |
| -------- | ------- | -------------- | --- | --------- | ----- | ----- | --- | --- | --- | --- |
Downloaded from https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103 by guest on 09 March 2026
| optimizes | the   | contrastive | objective | and     | prediction  | task |     |     |     |     |
| --------- | ----- | ----------- | --------- | ------- | ----------- | ---- | --- | --- | --- | --- |
| jointly.  | MVGCN | obtains     |           | optimal | performance | on   |     |     |     |     |
AUPR,whichvalidatesthatcontrastivelearningcanmit-
igatetheimpactofdataimbalance.Surprisingly,insome
cases, the performance of NIMCGCN, LAGCN, and Figure2.TheRecallsofallcomparedapproachesobtainedin10-CV.
MVGCNisworsethanthatofBNNRandDRIMC.The
reasonmightbethatNIMCGCNignorestheinteraction themost.WhenonlyusingtheDDAviewencoder,themodel
ofnodesinheterogeneousnetworks,andLAGCNindis- achieves a suboptimal performance, which is much higher
criminately mixes the network topology information of than the performance of the SMGCL without co-contrastive
learningtaskonboththethreedatasets.Thiscandemonstrate
| different | domains | (i.e. | drug | and disease | domains), | and |     |     |     |     |
| --------- | ------- | ----- | ---- | ----------- | --------- | --- | --- | --- | --- | --- |
theeffectivenessofmodelingtheinteractionbetweenneighbor
| MVGCN | does | not select | the | nearest | neighbor | of each |     |     |     |     |
| ----- | ---- | ---------- | --- | ------- | -------- | ------- | --- | --- | --- | --- |
nodetoconstructthesimilarityview,whichintroducesa nodes.Bycomparison,onlyusingthesimilarityviewencoder
lotofnoiseinformation. wouldleadtoahugeperformancedegradationonthreedata-
3) The AUROC obtained by SMGCL on Fdataset and sets. Surprisingly, removing the co-contrastive learning task
|          |       |          |              |     |          |       | and using | the sum of drug/disease | embeddings | on two views |
| -------- | ----- | -------- | ------------ | --- | -------- | ----- | --------- | ----------------------- | ---------- | ------------ |
| Cdataset | shows | the best | performance, |     | on LRSSL | shows |           |                         |            |              |
great performance. Compared with GRGMF and toobtainthefinalembeddingdonotachievesuboptimalper-
MVGCN,theaverageAUROCofSMGCLincreasedby formance.Thisprovesthatcontrastivelearningcanautomati-
15.54%and9.55%,respectively.Moreover,inthecon- callyminelabels,soastomaximizeagreementbetweennodes
|         |           |                 |     |      |         |           | in different | view. According | to this ablation | study, we can |
| ------- | --------- | --------------- | --- | ---- | ------- | --------- | ------------ | --------------- | ---------------- | ------------- |
| text of | imbalance | classification, |     | AUPR | is also | an indis- |              |                 |                  |               |
pensable evaluation metric. Compared with BNNR and concludethatasuccessfulDDApredictionmodelshouldcon-
DRWBNCF,theaverageAUPRofSMGCLincreasedby sidernotonlytheinteractionbetweendrugsanddiseases,but
27.96%and12.98%,respectively.Toclarifytheadvan- also the relationship between drugs and drugs, diseases and
| tages of | SMGCL, | more  | detailed         | comparison |         | between | diseases. |     |     |     |
| -------- | ------ | ----- | ---------------- | ---------- | ------- | ------- | --------- | --- | --- | --- |
| SMGCL    | and    | MVGCN | in Supplementary |            | Section | S2.4.   |           |     |     |     |
3.4Casestudy:approveddrugsforAlzheimer’s
| Benchmarking |     | comparison | results | show | that | SMGCL |     |     |     |     |
| ------------ | --- | ---------- | ------- | ---- | ---- | ----- | --- | --- | --- | --- |
diseasedeterminedbycalculation
| improves | the | comprehensive |     | prediction | performance |     |     |     |     |     |
| -------- | --- | ------------- | --- | ---------- | ----------- | --- | --- | --- | --- | --- |
thankstocombiningtheinformationoftheknownDDA We conduct a case study for the neurodegenerative disease
Alzheimer’sdisease(AD),forwhichtherearecurrentlynoef-
| is co-trained |     | with the | neighborhood | and | neighborhood |     |     |     |     |     |
| ------------- | --- | -------- | ------------ | --- | ------------ | --- | --- | --- | --- | --- |
interaction information of drugs and diseases under the fective treatments, in order to further evaluate the predictive
frameworkofcontrastivelearning. capabilityofSMGCL.AlloftheknownDDAsintheFdataset
areusedasthetrainingsetandtheunknownDDAsareused
3.3Modelablation as the candidate set when trying to find possible AD drugs.
To evaluate the rationality of design sub-modules in our Once the SMGCLpredicts the probabilityof interaction ofa
SMGCL framework, we consider three model variants as givendiseasewithalldrugcandidates,werankthecandidates
follows: according to that predicted probability, so that the top-
rankeddrugisthemostlikelytotreatthedisease.
1) SMGCLwithoutDDAviewencoder(w/o-DE):Weonly
|     |     |     |     |     |     |     | We focus | on the top 15 potential | candidates | for AD pre- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------------------- | ---------- | ----------- |
usethesimilarityviewstomodeldrugsanddiseases,re- dicted by SMGCL in Table 3. For each drug, we show the
movingthegraphco-contrastivelearning. DrugBank ID, canonical name and literature-reported evi-
2) SMGCL without similarity view encoder (w/o-AE): We dence,whichcheckthepredictedDDAs.Then,weselectthree
onlyusetheDDAviewtomodeldrugsanddiseases,re- drugs in Table 3 to describe them in detail. Amantadine has
movingthesimilarityviews,interaction-awaresimilarity antiviral, anti-Parkinson’s, and anti-pain activities. By pro-
views,andthegraphco-contrastivelearning. moting dopamine release from striatal dopaminergic nerve
3) SMGCL without co-contrastive learning task (w/o-CL): terminalsandpreventingitspre-synapticreuptake,ithasanti-
We remove the graph co-contrastive learning task and Parkinsonian actions. Furthermore, Erkulwater and Pillai
onlyusesimplesummingofdrug/diseaseembeddingson
(1989)haveprovedthatthementalstatusoftwoADpatients
twoviewstogetthefinalembedding. has obviously improved after treatment with amantadine.
Haloperidolisahighlyeffectivefirst-generationantipsychotic
As can be observed in Fig. 2, each component contributes drug and one of the most commonly used antipsychotics in
tothefinalperformance.TheDDAviewencodercontributes clinical practice today. Devanand et al. (1998) have

| SMGCL |     |     |     |     |     |     |     |     |     |     |     |     |     | 7   |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Funding
Table3.Thetop15SMGCL-predictedcandidatedrugsforAD.
Rank Drugname DrugBankIDs Evidence ThisworkwassupportedbytheIndustrialSupportProjectof
|               |     |         |     |                           |     |     |     | Gansu Colleges     |         | [No. 2022CYZC-11]; |       |            | the National  | Natural |
| ------------- | --- | ------- | --- | ------------------------- | --- | --- | --- | ------------------ | ------- | ------------------ | ----- | ---------- | ------------- | ------- |
| 1 Amantadine  |     | DB00915 |     | ErkulwaterandPillai(1989) |     |     |     |                    |         |                    |       |            |               |         |
|               |     |         |     |                           |     |     |     | Science Foundation |         | of                 | China | [61762078, | 61363058];    | Gansu   |
| 2 Ropinirole  |     | DB00268 |     | Shaughnessy(2006)         |     |     |     |                    |         |                    |       |            |               |         |
|               |     |         |     |                           |     |     |     | Natural            | Science | Foundation         |       | Project    | [21JR7RA114]; |         |
| 3 Haloperidol |     | DB00502 |     | Devanandetal.(1998)       |     |     |     |                    |         |                    |       |            |               |         |
4 Isoprenaline DB01064 Ohmetal.(1991) Northwest Normal University Young Teachers Research
5 Carbidopa DB00190 Meyeretal.(1977) Capacity Promotion Plan [NWNU-LKQN2019-2]; Guangxi
6 Risperidone DB00734 Mintzeretal.(2006) KeyLaboratoryofTrustedSoftware(kx202303).
| 7 Scopolamine   |     | DB00747 |     | SanTang(2019)       |     |     |     |            |     |     |     |     |     |     |
| --------------- | --- | ------- | --- | ------------------- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
| 8 Phenobarbital |     | DB01174 |     | BrodieandKwan(2012) |     |     |     |            |     |     |     |     |     |     |
| 9 Dopamine      |     | DB00988 |     | Louzadaetal.(2004)  |     |     |     | References |     |     |     |     |     |     |
| 10 Phenytoin    |     | DB00252 |     | Dhikav(2006)        |     |     |     |            |     |     |     |     |     |     |
| 11 Benzatropine |     | DB00245 |     | NA                  |     |     |     |            |     |     |     |     |     |     |
BahdanauD,ChoK,BengioY.Neuralmachinetranslationbyjointly
12 Pramipexole DB00413 Bennettetal.(2016) Downloaded from https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103 by guest on 09 March 2026
learningtoalignandtranslate.In:3rdInternationalConferenceon
| 13 Terabenazine |     | DB04844 |     | Kilbournetal.(1993) |     |     |     |     |     |     |     |     |     |     |
| --------------- | --- | ------- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
LearningRepresentations,ICLRSanDiego,CA,USA,2015.
| 14 Carbamazepine |     | DB00564 |     | Olinetal.(2001) |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
BennettJ,BurnsJ,WelchPetal.SafetyandtolerabilityofR(þ)prami-
| 15 Ceftriaxone |     | DB01212 |     | Zumkehretal.(2015) |     |     |     |        |                     |     |             |     |          |                  |
| -------------- | --- | ------- | --- | ------------------ | --- | --- | --- | ------ | ------------------- | --- | ----------- | --- | -------- | ---------------- |
|                |     |         |     |                    |     |     |     | pexole | in mild-to-moderate |     | Alzheimer’s |     | disease. | J Alzheimers Dis |
Theunderlineindicatesthatwehavemadeadetailedanalysisand 2016;49:1179–87.
introductionofthesedrugsinthefollowing. BrodieMJ,KwanP.Currentpositionofphenobarbitalinepilepsyand
itsfuture.Epilepsia2012;53:40–6.
ChenT,KornblithS,NorouziMetal.Asimpleframeworkforcontras-
| conducted | an  | experiment | on the | efficacy | and | side effects | of  |                 |     |                        |     |     |             |             |
| --------- | --- | ---------- | ------ | -------- | --- | ------------ | --- | --------------- | --- | ---------------------- | --- | --- | ----------- | ----------- |
|           |     |            |        |          |     |              |     | tive learningof |     | visualrepresentations. |     | In: | Proceedings | of the 37th |
haloperidolandplacebointhetreatmentofpsychosisanddis-
InternationalConferenceonMachineLearning,ICML,Vol.119of
| ruptive behavior |      | in patients | with | AD.    | Ultimately,  | the results |       |             |     |         |          |           |         |            |
| ---------------- | ---- | ----------- | ---- | ------ | ------------ | ----------- | ----- | ----------- | --- | ------- | -------- | --------- | ------- | ---------- |
|                  |      |             |      |        |              |             |       | Proceedings | of  | Machine | Learning | Research, | Virtual | Event, pp. |
| have shown       | that | haloperidol | at   | a dose | of 2–3mg/day |             | had a |             |     |         |          |           |         |            |
1597–607.PMLR,2020.
goodtherapeuticeffect.Carbidopaisthelevorotatoryisomer
|     |     |     |     |     |     |     |     | Devanand | D, Marder | K, Michaels |     | KS et al. | A randomized, | placebo- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ----------- | --- | --------- | ------------- | -------- |
ofasynthetichydrazinederivativeoftheneurotransmitterdo- controlled dose-comparison trial of haloperidol for psychosis and
pamine. Meyer et al. (1977) have performed serial clinical disruptivebehaviorsinAlzheimer’sdisease.AmJPsychiatry1998;
| assessments | and | neuropsychological |     | measures |     | of functioning |     | 155:1512–20. |     |     |     |     |     |     |
| ----------- | --- | ------------------ | --- | -------- | --- | -------------- | --- | ------------ | --- | --- | --- | --- | --- | --- |
in10patientswithseveredementiaconsistingofADormulti- DhikavV.CanphenytoinpreventAlzheimer’sdisease?MedHypotheses
| infarct dementia |      | (MID) or     | both, | who have | taken   | Carbidopa. |     | 2006;67:725–8. |           |               |     |     |               |             |
| ---------------- | ---- | ------------ | ----- | -------- | ------- | ---------- | --- | -------------- | --------- | ------------- | --- | --- | ------------- | ----------- |
|                  |      |              |       |          |         |            |     | Erkulwater     | S, Pillai | R. Amantadine |     | and | the end-stage | dementia of |
| The results      | have | demonstrated |       | that one | patient | with AD    | þ   |                |           |               |     |     |               |             |
Alzheimer’stype.SouthMedJ1989;82:550–4.
MIDdemonstratedclinicalandpsychologicalimprovement.
FuH,HuangF,LiuXetal.MVGCN:dataintegrationthroughmulti-
Overall,avarietyofevidencefromclinicaltrialsandother
viewgraphconvolutionalnetworkforpredictinglinksinbiomedical
literaturedatahavevalidated14ofthetop15predicteddrugs
bipartitenetworks.Bioinformatics2022;38:426–34.
(93%successrate),orderedbyconfidencescores. GlorotX,BengioY.Understandingthedifficultyoftrainingdeepfeed-
|     |     |     |     |     |     |     |     | forward | neural | networks. | In: | Proceedings | of  | the Thirteenth |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------ | --------- | --- | ----------- | --- | -------------- |
4Conclusion International Conference on Artificial Intelligence and Statistics,
AISTATS,pp.249–56,2010.
Inthisstudy,welookintothepotentialofGCLtoaddressthe
GottliebA,SteinGY,RuppinEetal.PREDICT:amethodforinferring
shortcomingsofthetraditionalDDAprediction.Inparticular, novel drug indications with application to personalized medicine.
an end-to-end SMGCL model is suggested to tap candidate MolSystBiol2011;7:496.
drugsfordiseases.Tobespecific,welearntherepresentation Howard AG. Some improvements on deep convolutional neural net-
ofdrugsanddiseases onthree relevantviewsandthen intro- workbasedimageclassification.In:BengioY,LeCunY(eds),2nd
duce a co-contrastive learning strategy that can sample posi- InternationalConferenceonLearningRepresentations,ICLR2014,
Banff,AB,Canada,2014.
| tive samples | and | dig hard | negative |     | samples | to generate |     |     |     |     |     |     |     |     |
| ------------ | --- | -------- | -------- | --- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
HuangC,ChenJ,XiaLetal.Graph-enhancedmulti-tasklearningof
| accurate | node | representations. | Finally, |     | experiments | on  | three |             |            |          |     |                   |     |                 |
| -------- | ---- | ---------------- | -------- | --- | ----------- | --- | ----- | ----------- | ---------- | -------- | --- | ----------------- | --- | --------------- |
|          |      |                  |          |     |             |     |       | multi-level | transition | dynamics |     | for session-based |     | recommendation. |
benchmarkdatasetsjustifytheadvantagesofourproposalre-
In:35thAAAIConferenceonArtificialIntelligence,AAAI,Virtual
| garding | DDA prediction. |     | The reliability |     | of the | newly discov- |     |     |     |     |     |     |     |     |
| ------- | --------------- | --- | --------------- | --- | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Event,pp.4123–30.AAAIPress,2021.
eredDDAshasbeensupportedbycasestudy.
KilbournMR,DaSilvaJN,FreyKAetal.Invivoimagingofvesicular
Since the task of DDA prediction is closely related to bio- monoaminetransportersinhumanbrainusing[11C]tetrabenazine
logical safety and human health. It is crucial to design a rea- andpositronemissiontomography.JNeurochem1993;60:2315–8.
sonable negative sampling strategy for constructing a robust KingmaDP,BaJ.Adam:amethodforstochasticoptimization.In:3rd
DDA prediction model. In future work, we will consider de- InternationalConferenceonLearningRepresentations,ICLR,San
Diego,CA,USA,2015.
| veloping | a proper | negative | sampling |     | strategy | for the | DDA |     |     |     |     |     |     |     |
| -------- | -------- | -------- | -------- | --- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
KipfTN,WellingM.Semi-supervisedclassificationwithgraphconvolu-
predictiontaskandanalyzetheperformanceimprovementof
|     |     |     |     |     |     |     |     | tional | networks. | In: 5th | International |     | Conference | on Learning |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --------- | ------- | ------------- | --- | ---------- | ----------- |
thenegativesamplingstrategyondifferentSOTAmodels.
Representations,ICLR,Toulon,France.OpenReview.net,2017.
|     |     |     |     |     |     |     |     | Li J, Zhang | S, Liu | T et al. | Neural | inductive | matrix | completion with |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | -------- | ------ | --------- | ------ | --------------- |
Supplementarydata graph convolutional networks for miRNA-disease association pre-
diction.Bioinformatics2020;36:2538–46.
SupplementarydataareavailableatBioinformaticsonline. LiangX,ZhangP,YanLetal.LRSSL:predictandinterpretdrug–dis-
|                    |     |     |     |     |     |     |     | ease associations                       |     | based | on data | integration | using | sparse subspace |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | ----- | ------- | ----------- | ----- | --------------- |
| Conflictofinterest |     |     |     |     |     |     |     | learning.Bioinformatics2017;33:1187–96. |     |       |         |             |       |                 |
LouzadaPR,LimaACP,Mendonca-SilvaDLetal.Taurinepreventsthe
Nonedeclared.
|     |     |     |     |     |     |     |     | neurotoxicity |     | of b-amyloid |     | and glutamate | receptor | agonists: |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------ | --- | ------------- | -------- | --------- |

8 Gaoetal.
activation of GABA receptors and possible implications for Computer Vision and Pattern Recognition, CVPR, pp. 3733–42.
Alzheimer’s disease and other neurological disorders. FASEB J Salt Lake City, UT, USA: Computer Vision Foundation/IEEE
2004;18:511–8. ComputerSociety,2018.
LuoH,WangJ,LiMetal.Drugrepositioningbasedoncomprehensive YangM,LuoH,LiYetal.Drugrepositioningbasedonboundednu-
similaritymeasuresandbi-randomwalkalgorithm.Bioinformatics clearnormregularization.Bioinformatics2019;35:i455–63.
2016;32:2664–71. YouY,ChenT,SuiYetal.Graphcontrastivelearningwithaugmenta-
MengY,LuC,JinMetal.Aweightedbilinearneuralcollaborativefil- tions. In: Advances in Neural Information Processing Systems 33:
tering approach for drug repositioning. Brief Bioinform 2022;23: Annual Conference on Neural Information Processing Systems,
bbab581. NeurIPS,Vol.33,pp.5812–23.2020.
Meyer JS, Welch K, Deshmukh V et al. Neurotransmitter precursor Yu Z, Huang F, Zhao X et al. Predicting drug–disease associations
amino acids in the treatment of multi-infarct dementia and through layer attention graph convolutional network. Brief
Alzheimer’sdisease.JAmGeriatrSoc1977;25:289–98. Bioinform2021;22:bbaa243.
MintzerJ,GreenspanA,CaersIetal.Risperidoneinthetreatmentof ZhangW,XuH,LiXetal.DRIMC:animproveddrugrepositioning
psychosis of Alzheimer disease: results from a prospective clinical approach using Bayesian inductive matrix completion.
trial.AmJGeriatrPsychiatry2006;14:280–91. Bioinformatics2020a;36:2839–47.
OhmTG,BohlJ,LemmerB.Reducedbasalandstimulated(isoprena- ZhangW,YueX,LinWetal.Predictingdrug-diseaseassociationsby
line,Gpp(NH)p,forskolin)adenylatecyclaseactivityinAlzheimer’s using similarity constrained matrix factorization. BMC
diseasecorrelatedwithhistopathologicalchanges.BrainRes1991; Bioinformatics2018;19:1–12.
540:229–36. ZhangZ-C,ZhangX-F,WuMetal.Agraphregularizedgeneralized
OlinJT,FoxLS,PawluczykSetal.Apilotrandomizedtrialofcarba- matrixfactorizationmodelforpredictinglinksinbiomedicalbipar-
mazepineforbehavioralsymptomsintreatment-resistantoutpatients titenetworks.Bioinformatics2020b;36:3474–81.
withAlzheimerdisease.AmJGeriatrPsychiatry2001;9:400–5. ZhaoC,LiuS,HuangFetal.CSGNN:contrastiveself-supervisedgraph
San Tang K. The cellular and molecular processes associated with neuralnetworkformolecularinteractionprediction.In:Proceedings
scopolamine-induced memory deficit: a model of Alzheimer’s bio- ofthe30thInternationalJointConferenceonArtificialIntelligence,
markers.LifeSci2019;233:116695. IJCAI,VirtualEvent,pp.3756–63.Montreal,Canada,2021.
ShaughnessyAF.Ropinirolemaybeeffectiveforrestlesslegssyndrome. ZhuH,FengF,HeXetal.Bilineargraphneuralnetworkwithneighbor
AmFamPhysician2006;73:2217. interactions. In: Proceedings of the 29th International Joint
TongH,FaloutsosC,PanJ.Fastrandomwalkwithrestartanditsappli- ConferenceonArtificialIntelligence,IJCAI,pp.1452–8.IJCAI.org,
cations.In:Proceedingsofthe 6thIEEEInternationalConference 2020.
on Data Mining ICDM, pp. 613–22. Hong Kong, China: IEEE ZumkehrJ,Rodriguez-OrtizCJ,ChengDetal.Ceftriaxoneameliorates
ComputerSociety,2006. taupathologyandcognitivedeclineviarestorationofglialglutamate
Wu Z, XiongY, YuSX etal. Unsupervisedfeature learningvia non- transporter in a mouse model of Alzheimer’s disease. Neurobiol
parametric instance discrimination. In: 2018 IEEE Conference on Aging2015;36:2260–71.
Downloaded
from
https://academic.oup.com/bioinformatics/article/39/6/btad357/7188103
by
guest
on
09
March
2026