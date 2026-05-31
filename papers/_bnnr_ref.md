Bioinformatics,35,2019,i455–i463
doi:10.1093/bioinformatics/btz331
ISMB/ECCB2019
Drug repositioning based on bounded nuclear
norm regularization
Mengyun Yang1,2, HuiminLuo1, YaohangLi3and JianxinWang 1,*
1School of Computer Science and Engineering, Central South University, Changsha 410083, 2Provincial Key
Laboratory of Informational Service for Rural Area of Southwestern Hunan, Shaoyang University, Shaoyang
422000,Chinaand3DepartmentofComputerScience,OldDominionUniversity,Norfolk,VA23529,USA
*Towhomcorrespondenceshouldbeaddressed.
Abstract
Motivation: Computational drug repositioning is a cost-effective strategy to identify novel indica-
tionsforexistingdrugs.Drugrepositioningisoftenmodeledasarecommendationsystemproblem.
Takingadvantageoftheknowndrug–diseaseassociations,theobjectiveoftherecommendationsys-
temistoidentifynewtreatmentsbyfillingouttheunknownentriesinthedrug–diseaseassociation
matrix,whichisknownasmatrixcompletion.Underpinnedbythefactthatcommonmolecularpath-
wayscontributetomanydifferentdiseases,therecommendationsystemassumesthattheunderly-
ing latent factors determining drug–disease associations are highly correlated. In other words, the
drug–disease matrix to be completed is low-rank. Accordingly, matrix completion algorithms effi-
ciently constructing low-rank drug–disease matrix approximations consistent with known associa-
tionscanbeofimmensehelpindiscoveringthenoveldrug–diseaseassociations.
Results:Inthisarticle,weproposetouseaboundednuclearnormregularization(BNNR)methodto
complete the drug–disease matrix under the low-rank assumption. Instead of strictly fitting the
knownelements,BNNRisdesignedtotoleratethenoisydrug–druganddisease–diseasesimilarities
by incorporating a regularization term to balance the approximation error and the rank properties.
Moreover,additionalconstraintsareincorporatedintoBNNRtoensurethatallpredictedmatrixentry
values are within the specific interval. BNNR is carried out on an adjacency matrix of a heteroge-
neous drug–disease network, which integrates the drug–drug, drug–disease and disease–disease
networks. It not onlymakes full use of available drugs, diseases and their association information,
but alsoiscapable of dealing with cold start naturally.Our computational resultsshowthat BNNR
yieldshigherdrug–diseaseassociationpredictionaccuracythanthecurrentstate-of-the-artmethods.
Themostsignificantgainisinpredictionprecisionmeasuredasthefractionofthepositivepredic-
tionsthat are truly positive, which is particularly useful in drug design practice. Cases studies also
confirmtheaccuracyandreliabilityofBNNR.
Availability and implementation: The code of BNNR is freely available at https://github.com/
BioinformaticsCSU/BNNR.
Contact:jxwang@mail.csu.edu.cn
Supplementaryinformation:SupplementarydataareavailableatBioinformaticsonline.
1Introduction owned safety, efficacy and toleration data after various tests and
clinicaltrials.Theprocessofidentifyingnewapplicationsforexist-
Theprocessofnewdrugdiscoveryistime-consumingandtremen- ingdrugsisknownasdrugrepositioning.Infact,somesuccessfully
douslyexpensive(Chongetal.,2007).Ithasbeenshowedthatthe repositioned drugs, such as sildenafil, raloxifene and thalidomide,
averagetimeofdevelopinganewdrugismorethan13.5yearsand havegeneratedgenerousrevenuesfortheirpatentholdersorcompa-
thecostexceeds$1.8billiondollars(Pauletal.,2010).Discovering nies.Therefore,drugrepositioningisaneffectivestrategyfordevel-
new and reliable indications for commercialized drugs allows the opingnewdrugs.
pharmaceuticalindustryandtheresearchcommunitytoreducetime Computationaldrugrepositioninghasattractedincreasingattention,
andcosts, because the existing commercialized drugs have already sincemanualinvestigationistime-consuming.Withthedevelopmentof
VCTheAuthor(s)2019.PublishedbyOxfordUniversityPress. i455
ThisisanOpenAccessarticledistributedunderthetermsoftheCreativeCommonsAttributionNon-CommercialLicense(http://creativecommons.org/licenses/by-nc/4.0/),
whichpermitsnon-commercialre-use,distribution,andreproductioninanymedium,providedtheoriginalworkisproperlycited.Forcommercialre-use,pleasecontact
journals.permissions@oup.com
Downloaded
from
https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141
by
guest
on
29
April
2020

| i456 |     |     |     |     |     |     |     |     | M.Yangetal. |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- |
highthroughputtechnologyandcontinuouslyupdatingdatabases,quite and develop a bounded nuclear norm regularization (BNNR)
a few computational approaches have been proposed, including methodtoaddressthisproblem.Firstofall,weconstructaheteroge-
network-basedanalysis,machinelearning,textminingandsemanticin- neous drug–disease network, which is composed of drug–drug,
ferenceapproaches.Thenetwork-basedmethodsarepopularandfun- drug–disease and disease–disease sub-networks. Then, BNNR is
damentalfordrugrepositioning.Basedonanetworkofdrugs,diseases implementedtorecoverthemissingentriesintheadjacencymatrix
and targets (proteins), Martinez et al. (2015) proposed an approach ofthisheterogeneousnetworkwhiletoleratingthepotentialnoisein
namedDrugNettopredictnewuseforexistingdrugs.DrugNetcanper- drug–druganddisease–diseasesimilaritiescalculations.Finally,we
formbothdrug–diseaseanddisease–drugprioritizationbypropagating evaluate the performance of BNNR on various datasets and com-
informationintheheterogeneousnetwork.Gottliebetal.(2011)inte- pareitwithseveralstate-of-the-artmethods.Ourresultsshowthat
grateddrugsimilaritiesanddiseasesimilaritiestoobtainprimaryfea- ourapproachhassuperiorcapabilityofpredictinghiddendrug–dis-
turestosupportacomputationalapproachcalledPREDICTtoidentify ease associations. The main contributions of our BNNR model
unknowndrug–diseaseassociations.Wangetal.(2013)constructeda include: Downloaded from https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141 by guest on 29 April 2020
heterogeneousdrug–targetgraph,whichcontainsintra-similarityinfor-
•
mation and drug–target association information. Based on the guilt- BNNR performsnoisymatrix completion by incorporating nu-
clearnormregularization,whicheffectivelyaddressesoverfitting
| by-association | principle, | heterogeneous |     | graph based | inference (HGBI) |     |     |     |     |
| -------------- | ---------- | ------------- | --- | ----------- | ---------------- | --- | --- | --- | --- |
andleadstobetterimprovedaccuracyasshowninourresults;
algorithm(Wangetal.,2013)wasproposedtopredictnewdrug–target
• OurBNNRmodelincorporatesarangeconstraint,whichenfor-
associations.HGBIisalsousedforpredictingdrug–diseaseassociations
cesallpredictedmatrixentryvalueswithinthespecificinterval;
(Wangetal.,2014).Luoetal.(2016)exploitedtheavailableinforma-
•
tionofdrug–diseaseassociationstoenhancedrugsimilarityanddisease OurBNNRmodelisabletodealwithnoisydataefficiently;and
• Anefficientiterativeschemeisdesignedtonumericallysolvethe
| similarity. | The MBiRW | algorithm, | which | used some | comprehensive |     |     |     |     |
| ----------- | --------- | ---------- | ----- | --------- | ------------- | --- | --- | --- | --- |
BNNRmodel.
similaritymeasuresandBi-RandomWalk(BiRW)algorithm,isimple-
mentedonthedrug–diseaseheterogeneousnetworktopredictpotential
drug–diseaseassociations.
2Materialsandmethods
| Matrix | factorization | and | matrix | completion | techniques have |     |     |     |     |
| ------ | ------------- | --- | ------ | ---------- | --------------- | --- | --- | --- | --- |
Inthissection,wedescribetheBNNRmodeltopredictthepotential
beenappliedtodrugrepositioninginrecentyears.Daietal.(2015)
incorporatedtheinteractionnetworkofgenesanddevelopedama- indicationsforexistingdrugs,whichisorganizedasfollows.First,
trix factorization model. Taking advantage of the information in wedescribethedatasetsusedinthisstudy.Then,wedepictthecon-
genesnetwork,theassociationbetweendruganddiseasecanbepre- struction of the drug–disease heterogeneous network and its adja-
dictedandnewindicationsforknowndrugscanbeobtained.Luo cencymatrixtobecompleted.Finally,wepresenttheBNNRmodel,
solvedbyalternatingdirectionmethodofmultipliers(ADMM),to
| et al. (2018) | constructed | a   | heterogeneous | network | by integrating |     |     |     |     |
| ------------- | ----------- | --- | ------------- | ------- | -------------- | --- | --- | --- | --- |
drug–drugnetwork,disease–diseasenetworkanddrug–diseaseasso- fillouttheunknownassociationsbetweendrugsanddiseases.The
ciationnetwork,andthenR4SVD(LiandYu,2017)wasemployed overallworkflowofBNNRisillustratedinFigure1.
| to efficientlycomputethe |          | dominantsingularvaluesandthe |                 |         | corre-       |     |     |     |     |
| ------------------------ | -------- | ---------------------------- | --------------- | ------- | ------------ | --- | --- | --- | --- |
| sponding                 | singular | vectors of                   | the association | matrix. | Based on the |     |     |     |     |
2.1Datasets
Singular Value Thresholding (SVT) algorithm (Cai et al., 2010), a Weuse the gold standarddatasetto predict newdrug indications,
DrugRepositioningRecommendationSystem(DRRS)hasbeenpro- whichisobtainedfrom(Gottliebetal.,2011)collectingcomprehen-
posedtorankthepotentialassociationsbetweendrugsanddiseases
|               |          |              |             |            |                    | sive associations | frommultipledata   | sources.There | are 593drugs,       |
| ------------- | -------- | ------------ | ----------- | ---------- | ------------------ | ----------------- | ------------------ | ------------- | ------------------- |
| by completing | the      | drug–disease | association | matrix.    | In fact, the       |                   |                    |               |                     |
|               |          |              |             |            |                    | 313 diseases      | and 1933 validated | drug–disease  | associations. Drugs |
| methods       | based on | random walks | are         | equivalent | to certain special |                   |                    |               |                     |
arecollectedfromtheDrugBankdatabase(Wishartetal.,2006)and
cases of those using matrix completions. For example, MBiRW is diseases are extracted from the Online Mendelian Inheritance in
equivalent to finding the eigenvector with respect to the largest Man(OMIM)dataset(Adaetal.,2002).
| eigenvalue | of the | association | matrix. | However, | the above matrix |     |     |     |     |
| ---------- | ------ | ----------- | ------- | -------- | ---------------- | --- | --- | --- | --- |
completionalgorithmsareoperatedinanoiselesssetting,assuming
thatthedrug–diseaseassociationsarecorrectlyderivedandthedis-
| ease–disease | as well | as drug–drug | similarities | are | accurately meas- |     |     |     |     |
| ------------ | ------- | ------------ | ------------ | --- | ---------------- | --- | --- | --- | --- |
ured.Butinreality,drugsanddiseasesvaryinmanyaspectsanditis
difficulttoconstructasinglemeasuretopreciselydescribethesimi-
larityrelationshipamongdrugsordiseases.Occasionally,suchsimi-
larityismisleading.Forexample,adiseasecausedbybacteriamay
havehighlysimilarsymptomsasonecausedbyvirus,whichshould
| be treated  | by completely | different  | drugs. | Moreover,           | in the matrix |     |     |     |     |
| ----------- | ------------- | ---------- | ------ | ------------------- | ------------- | --- | --- | --- | --- |
| completions | algorithms,   | typically, | 1’s    | in the drug–disease | associ-       |     |     |     |     |
ationmatrixdenoteknowndrug–diseaseassociationswhile0’srep-
resenttheunknowns.Thepredictedvaluesareexpectedtobewithin
therangeof[0,1],indicatingthelikelinessofthepredictedassocia-
| tions. However, | the | above | matrix | factorization | and completion |     |     |     |     |
| --------------- | --- | ----- | ------ | ------------- | -------------- | --- | --- | --- | --- |
approachesareunabletoavoidthesituationsthatthepredictedval-
uesfalloutofthe[0,1]range,whichbringsdifficultyinbiological
interpretation.
Fig.1.TheoverallworkflowofBNNR.(a)Drug–drugnetworkanditssimilarity
Inthisstudy,assumingthatsimilardrugssharethesimilarmo- matrix.(b)Disease–diseasenetworkanditssimilaritymatrix.(c)Drug–dis-
lecularpathwaytotreatsimilardiseases,weconsidertheprediction easeassociationnetworkanditsassociationmatrix.(d)Theheterogeneous
of drug–disease association as a noisy matrix completion problem drug–diseasenetworkanditsadjacencymatrix.(e)ThemodelofBNNR

| DrugrepositioningbasedonBNNR |     |     |     |     |     |     |     |     |     |     | i457 |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
m inkXk
The similarities between drugs are calculated by the Chemical (cid:4)
|     |     |     |     |     |     |     |     |     | X   |     | (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
DevelopmentKit(CDK)(Steinbecketal.,2003)accordingtothechem-
|                 |     |          |                  |           |            |     |     | s:t:PX | ðXÞ¼PX | ðMÞ; |     |
| --------------- | --- | -------- | ---------------- | --------- | ---------- | --- | --- | ------ | ------ | ---- | --- |
| ical structures | of  | all drug | compounds in the | Canonical | Simplified |     |     |        |        |      |     |
MolecularInputLine-EntrySystem(SMILES)(Weininger,1988).We
|     |     |     |     |     |     | wherekXk | denotesthenuclearnormofX,whichisdefinedasthe |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | -------------------------------------------- | --- | --- | --- | --- |
(cid:4)
firstly download the Canonical SMILES format of all drugs from sum of all singular values of X. The nuclear norm minimization
DrugBank.Then,weutilizeCDKtocalculateabinaryfingerprintfor model is a convex optimization problem. Many algorithms have
each drug. Finally, the Tanimoto score (Tanimoto, 1958) measuring beendesignedtoprovidenumericalsolutionsfortheabovemodelor
thesimilarityofpairwisedrugsiscalculatedwithrespecttotheirchem-
|     |     |     |     |     |     | alternative | forms, including | the | fixed point | continuation | with ap- |
| --- | --- | --- | --- | --- | --- | ----------- | ---------------- | --- | ----------- | ------------ | -------- |
icalfingerprints,whichisintherangeof[0,1].
proximateSVD(FPCA)(Maetal.,2011),theacceleratedproximal
Disease–disease similarities are obtained from MimMiner (Van gradientalgorithm(APG)(Tohetal.,2010),theSVTalgorithm(Cai
Driel et al., 2006), which measure the number of appearance of etal.,2010)andtheADMM(Boydetal.,2011;Chenetal.,2012;
MeSH(medicalsubjectheadingsvocabulary)termsoftwodiseases Wen et al., 2010). Candes et al. (2013) showed that the solution Downloaded from https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141 by guest on 29 April 2020
inthemedicaldescriptionsobtainedfromtheOMIMdatabase.
obtainedbyoptimizingthenuclearnormisequivalenttotheoneby
rankminimizationundercertainconditions,minimizingthenuclear
| 2.2Constructionoftheheterogeneousnetwork |     |     |     |     |     | norm. |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
Weconstructaheterogeneousdrug–diseasenetwork,whichintegra- For predicting drug–disease associations, the elements in the
testhedrug–drug,disease–diseaseanddrug–diseaseassociationnet- drug similarity matrix A and disease similarity matrix A are
|     |     |     |     |     |     |     |     | RR  |     |     | DD  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
works.Let R¼fr 1 ;r 2 ;:::;r mg and D¼fd 1 ;d 2 ;:::;d ng denote a set withintheintervalof[0,1].Theelementsintheassociationmatrix
ofmdrugsandndiseases,respectively.Forthedrug–drugnetwork, A RD are either 0 or 1. As a result, the predicted values in the un-
theedgebetweentwodrugsisweightedbythepairwisedrugsimi- knownentriesareexpectedtobeintheintervalof[0,1],wherea
larityvalue.Similarly,theedgebetweentwodiseasesisweightedby predictedvaluecloserto1indicatesthatthisislikelytobeanindica-
thepairwisediseasesimilarityvalue.Then,thedrug–diseaseassoci- tion and vice versa. Nevertheless, in the above matrix completion
ation network is treated as a bipartite graph G(R, D, E), where models(1),theentriesinthecompletedmatrixcanbeanyrealvalue
EðGÞ¼fe ijg(cid:2)R(cid:3)D contains edges representing known associa- in((cid:5)1,þ1).Apredictedvalueoutoftheinterval[0,1]ismean-
tionsbetweendrugr anddiseased.Inthisheterogeneousdrug–dis- ingless in the application context. Hence, it is important to add a
|     |     | i   | j   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
easenetwork,drug–drugnetworkanddisease–diseasenetworkare boundconstrainttothematrixcompletionmodeltoensurethatthe
connected by drug–disease associations. Figure 1a–d illustratesthe uncoveredmissingelementsarewithintheintervalof[0,1].
constructionoftheheterogeneousnetwork. Moreover,sincetheremaybealot‘noise’inthedruganddisease
The adjacency matrix of the drug–disease heterogeneous net- data, particularly when measuring the drug–drug and disease–
workisthendefinedas: disease similarities, the drug repositioning model should effective
toleratethepotentialnoise.Amatrixcompletionmodeltotolerate
|     |     |     | (cid:2) AT (cid:3) |     |     |          |     |     |     |     |     |
| --- | --- | --- | ------------------ | --- | --- | -------- | --- | --- | --- | --- | --- |
|     |     | M¼  | A RR DR ;          |     |     | noiseis: |     |     |     |     |     |
|     |     |     | A DR A DD          |     |     |          |     |     |     |     |     |
minkXk
wherethesub-matricesA andA denotetheadjacencymatrices X (cid:4)
|     |     | RR  | DD  |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ofdrugnetworkanddiseasenetworkandtheirweightsaresetasthe s:t:jjPXðXÞ(cid:5)PXðMÞjj (cid:6) (cid:2);
F
pairwisedruganddiseasesimilarities,respectively,inrange[0,1].
where(cid:2)measuresthenoiselevel.However,forthismodelwiththe
| A andA | aredensewhichincluderichcorrelationinformation |     |     |     |     |     |     |     |     |     |     |
| ------ | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
RR DD inequality constraint, choosing the appropriate parameter is chal-
amongdrugsanddiseases.Incontrast,duetothefactthatdrug–dis-
lenging,becausethenoiselevelisnotexplicitlyknown.Moreover,it
| ease associations | are | rare, | A is usually extremely |     | sparse, where |     |     |     |     |     |     |
| ----------------- | --- | ----- | ---------------------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
DR
1’s denote known drug–disease associations and 0’s correspond to is not straightforward to come up with an efficient solver for this
model.Therefore,werelaxtheconstraintsatisfactionmodelintoa
theunknowns.Afterall,ourgoalistofillouttheunknownelements
|         |               |        |              |              |         | regularization | model. | Introducing | the | soft regularization | term not |
| ------- | ------------- | ------ | ------------ | ------------ | ------- | -------------- | ------ | ----------- | --- | ------------------- | -------- |
| in A as | the predicted | scores | of potential | associations | between |                |        |             |     |                     |          |
DR only enables tolerance to the unknown noise (Chen et al., 2012;
drugsanddiseases.
Huetal.,2013;Maetal.,2011;Tohetal.,2010),butalsoprovides
computationalconvenience.
2.3BNNRforpredictingdrug–diseaseassociations
Puttingallpiecestogether,weproposeaBNNRmethod,which
Assumingalow-rankstructure,thegeneralmatrixcompletionprob-
minimizesthenuclearnormastheregularizationtermandensures
lem(Ramlatchanetal.,2018)tofilloutthemissingentriesisformu-
therecoveredmatrixelementswithinaspecificinterval.TheBNNR
latedas:
modelisdescribedasfollows:
|                            |     |        | minrankðXÞ           |         |         |     |         | a                 |                      |     |     |
| -------------------------- | --- | ------ | -------------------- | ------- | ------- | --- | ------- | ----------------- | -------------------- | --- | --- |
|                            |     |        | X                    |         |         |     | m inkXk | þ                 | kPXðXÞ(cid:5)PXðMÞk2 |     |     |
|                            |     | s:t:PX | ðXÞ¼PXðMÞ;           |         |         |     |         | (cid:4) 2         |                      | F   |     |
|                            |     |        |                      |         |         |     | X       |                   |                      |     | (2) |
|                            |     |        |                      |         |         |     | s:t:0   | (cid:6) X (cid:6) | 1;                   |     |     |
| where M2RðmþnÞ(cid:3)ðmþnÞ |     | is     | the given incomplete | matrix, | rank(.) |     |         |                   |                      |     |     |
denotestherankfunction,Xisasetcontainingindexpairs(i,j)of whereaisparameterbalancingthenuclearnormandtheerrorterm.
allknownentriesinMandPXistheprojectionoperatorontoX.
|     |     |     |     |     |     | Note that | we use 0 | (cid:6) X (cid:6) | 1 to denote | 0 (cid:6) X ij (cid:6) | 1 for all ele- |
| --- | --- | --- | --- | --- | --- | --------- | -------- | ----------------- | ----------- | ---------------------- | -------------- |
(cid:4) mentsinXthroughoutthispaper.Wederiveasimplebuteffective
|     |         |     | X ; ði;j Þ 2 | X   |     |                                     |     |     |     |     |     |
| --- | ------- | --- | ------------ | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- |
|     | ðPXðXÞÞ |     | ¼ ij         | :   |     | numericalschemeusingADMMtosolve(2). |     |     |     |     |     |
|     |         | ij  | 0 ; ð i; jÞ  | 62X |     |                                     |     |     |     |     |     |
Model(2)issolvedbyaniterativemethod.Startingfromtheini-
Unfortunately, the rank minimization problem is known to be tialsolution X 1¼PXðMÞ.Itisimportanttonoticethattheobject-
NP-hard. The rank minimization in the above matrix completion ivefunctionin(2)isconvex.ByintroducinganauxiliarymatrixW,
model is often relaxed to a nuclear norm minimization problem (2)canbeoptimizedusingtheADMMframeworkinthefollowing
| suchthat: |     |     |     |     |     | equivalentform. |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- |

i458 M.Yangetal.
a
m
X
inX (cid:4)þ
2
PXðWÞ(cid:5)PXðMÞ2
F 2
w
0
h
1
e
0
re
;M
D
a
sð
e
X
t
Þ
al
i
.
s
,2
th
0
e
11
s
)
in
d
g
e
u
fi
l
n
a
e
r
d
v
a
a
s
lue shrinkage operator (Cai et al.,
s:t: X¼W; (3)
0 (cid:6) W (cid:6) 1: DsðXÞ¼ X
ri(cid:8)s
ðr i(cid:5)sÞu
i
v
i
T;
i¼1
Accordingly,theaugmentedLagrangianfunctionbecomes
a
wherer
i
isthesingularvaluesofXwhichislargerthans,whileu
i
LðW;X;Y;a;bÞ¼kXk (cid:4) þ 2 kPXðWÞ(cid:5)PXðMÞk2 F and v i are the left and right singular vectors corresponding to r i ,
þTr (cid:5) YTðX(cid:5)WÞ (cid:6) þ b kX(cid:5)Wk2; (4) respectively.
2 F Compute Y :Finally,Y iscalculatedas
kþ1 kþ1
whereYistheLagrangemultiplierandb>0isthepenaltyparam- (cid:9) pffiffiffi (cid:10)
5þ1
eter. At the k-th iteration, BNNR requires alternatively Y kþ1 ¼Y k þcbðX kþ1 (cid:5)W kþ1 Þ; c2 0; 2 ; (10)
computing W ; X and Y .
kþ1 kþ1 kþ1
Compute W :WefixX andY tominimizeLðW;X ;Y ;a;bÞ wherecisthelearningrate,whichissetto1inthisstudyforsimpli-
kþ1 k k k k
for W .Weherebytakefulladvantageoftheinverseoperatortoob- city (Hu et al., 2013). Putting all pieces together, Algorithm 1
kþ1
tainanexactandclosed-formsolution. presentsaniterativeBNNRschemeforsolving(2).Basedontheas-
sumptionthat similardiseasestend to be treatedby similar drugs,
W ¼arg min LðW;X ;Y ;a;bÞ
kþ1 k k becauseofthecommonmolecularpathways,thereexistcertainlow-
0(cid:6)W(cid:6)1
¼a 0 rg (cid:6)W m (cid:6) in 1 2 a kPXðWÞ(cid:5)PXðMÞk2 F (5) r n a u n c k le s a t r ru n c o tu rm res o g f ov t e h r e ni t n a g rg d e r t ug m – a d t i r s i e x a , se B a N ss N o R cia r ti e o v n e s a . ls M t i h n e im lo iz w in - g ra t n h k e
þTr (cid:5) Y TðX (cid:5)WÞ (cid:6) þ b kX (cid:5)Wk2: structuresandprovidesawaytorecoverthemissingentries.After
k k 2 k F
supplying the adjacency matrix of the drug–disease heterogeneous
Here,W*istheoptimalsolutionofarg min LðW;X ;Y ;a;bÞ, networktoBNNR,wecanobtainanupdateddrug–diseaseassoci-
k k
ifandonlyif W ationmatrix A(cid:4) DR ,wheretheunknownentriesinA DR arefilledup.
TheentriesinA(cid:4) withpredictedvalues(scores)closeto1indicate
aP(cid:4)
X
ðPXðW(cid:4)Þ(cid:5)PXðMÞÞ(cid:5)Y
k
(cid:5)bðX
k
(cid:5)W(cid:4)Þ¼0 (6)
thepotentialdru
D
g
R
–diseaseassociations.
holds,whereP(cid:4)
X
denotestheadjointoperatorof PX.Then,aclosed-
formsolutionbecomes
W(cid:4)¼
(cid:7)
Iþ
b
aP(cid:4)
X
PX
(cid:8)(cid:5)1 (cid:9)1
b
Y
k
þ
b
a
P(cid:4)
X
PX ðMÞþX
k
(cid:10) Algorithm1.BNNRAlgorithm
(cid:9) a (cid:10)(cid:9)1 a (cid:10) Input: The drug similarity matrix A RR2Rm(cid:3)m, the disease
¼ I(cid:5) aþb P(cid:4) X PX b Y k þ b P(cid:4) X PX ðMÞþX k similaritymatrix A DD2Rn(cid:3)n,thedrug–diseaseassoci-
(cid:9)1 a (cid:10)
(7) ationmatrix A DR2Rn(cid:3)m,parametersaandb.
¼ b Y k þ b PX ðMÞþX k Outpu " t:Predictedas # sociationmatrix A(cid:4) DR .
A A T
a (cid:9)1 a (cid:10) M A RR A DR ;
(cid:5)
aþb
PX
b
Y
k
þ
b
PX ðMÞþX
k
;
initialize
D
X
R
1¼P
D
X
D
ðMÞ;W 1¼X 1 ;Y 1¼X 1 ;c¼1; //X is a set
ofindicesofallknownentriesinM.
whereIistheidentityoperator.ðIþ
b
aP(cid:4)
X
PXÞ(cid:5)1denotestheinverse
k 1;
operatorofðIþ
b
aP(cid:4)
X
PXÞandisequaltoI(cid:5)
aþ
a
b
P(cid:4)
X
PX (Yangand
repeat
t i Y n o u t [ e a 0 r n v , , a 1 l 2 ] ½ 0 s 0 u 1 ; c 2 1 h ) (cid:7) . t c h I o t a n ’s t st w ra o i r n t t h ,w no e t l i i n m g it th th a e t r P a (cid:4) X n P ge X o ¼ ft P he X. el C em on e s n i t d s e o ri f n W g k th þ1 e Y X W k k þ þ 1 1 Y D Q 1 b ½(cid:7)0 þ ; W 1(cid:7) c ð b k W þ ðX 1 (cid:4)Þ (cid:5) ; 1 b Y (cid:5) k (cid:8) W ; Þ;
kþ1 k kþ1 kþ1
W kþ1 ¼Q ½0;1(cid:7) ðW(cid:4)Þ; (8) k kþ1;
untilconvergence
whereQ ½0;1(cid:7) istheprojectionoperatordefinedas (cid:2) A(cid:4) RR A(cid:4) DR T(cid:3) W ;
A(cid:4) A(cid:4) k
8 DR DD
>< 1; W i (cid:4) j >1 returnA(cid:4) :
ðQ ½0;1(cid:7) ðW(cid:4)ÞÞ ij ¼ W i (cid:4) j ; 0 (cid:6) W i (cid:4) j (cid:6) 1: DR
>:0; W
i
(cid:4)
j
< 0
Compute X :Alternatively,wefixW andY tocompute X .
kþ1 kþ1 k kþ1
3Resultsanddiscussion
X kþ1¼argmin LðW
kþ1
;X;Y
k
;a;bÞ
X (cid:7) (cid:8) b 3.1Evaluationmetrics
¼argminkXk (cid:4) þTr Y k TY k TðX(cid:5)W kþ1Þ þ 2 kX(cid:5)W kþ1k2 F ToevaluatetheperformanceofBNNR,a10-foldcross-validationis
X
b(cid:11) (cid:11) (cid:9) 1 (cid:10)(cid:11) (cid:11) 2 (9) conducted to verify the candidate diseases for given drugs. All
¼arg X(cid:9) minkXk (cid:4) þ 1 2 (cid:11) (cid:11) (cid:10) X(cid:5) W kþ1(cid:5) b Y k (cid:11) (cid:11) F knowndrug–diseaseassociationsarerandomlydividedinto10ex-
¼D1 W kþ1(cid:5) b Y k ; clusivesubsetsofapproximatelyequalsize.Eachsubsetistreatedas
thetestingsetinturn,whiletheremainingninesubsetsareusedas
b
the training set. The 10-fold cross-validation is repeated 10 times
Downloaded
from
https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141
by
guest
on
29
April
2020

| DrugrepositioningbasedonBNNR |     |     |     |     |     |     |     |     |     |     |     | i459 |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
with random subset division and the average accuracy values are associationinformationduetobetterrobustness.BNNRhasseveral
showedasthefinalresults. distinctadvantagescomparedwithDRRS:First,BNNRcouldfitthe
After the association matrix of the drug–disease heterogeneous wholenetworkbetter.Sincethevaluesof similaritymatricescom-
networkiscompleted,thepredictedscoresofalldrug–diseaseasso- putedinsilicomayincludenoisyinformation,BNNRhasarelaxed
ciationsareobtained.Foreachdrug,thepredictedscoresofitsasso- penaltyfunctiontocopewithnoisyentries,whileDRRSattemptsto
ciationswiththediseasesarerankedindescendingorder.Thescore fitallentries.Second,BNNRhasmoreinterpretablepredictedval-
ofthecandidateassociationexceedingagiventhresholdisconsid- ues.Theboundedconstraintensuresthatallpredictedassociations
ered as a positive prediction; otherwise, negative. For increasing arewithin[0,1].Incontrast,thepredictedassociationscoresmay
threshold values, true positive rate (TPR) and false positive rate benegativeor>1inDRRS.Third,theregularizationtermbasedon
(FPR)willbecalculatedtogeneratethereceiver-operatingcharacter- nuclearnormisabletoaddressoverfittingeffectively.Thisenables
istic (ROC) curve. Precision and recall (equivalent to TPR) are ustodesignanappropriatestopcriterionforBNNRtodirectlyob-
obtainedtoplottheprecision–recall(PR)curve(Davisetal.,2006). taintheoptimalsolutionwithouttheneedofdesignatingapartof Downloaded from https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141 by guest on 29 April 2020
Meanwhile,duetothefactthatthetop-rankedresultsareofmost knowndrug–diseaseassociationsasthevalidationsettoidentifythe
| interest,thenumberofcorrectlyidentifieddrug–diseaseassociations |     |     |     |     |     |     | optimalrank. |     |     |     |     |     |
| --------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
using different thresholds will be illustrated. The area under the To ensure a fair comparison, the parameters in the compared
ROCcurve(AUC)andtop-rankedresultsarepresentedtocompare approaches are set to the default values according to the authors’
theoverallperformanceofBNNRwithavarietyofexistingmethods recommendation (HGBI: a¼0.4; MBiRW: a¼0.3, l¼2, r¼2;
inthisstudy. DRRS: s and d are two adaptive parameters) and cross-validation
(DrugNet:aischosenfrom{0.1,0.2,...,0.9}).Theoverallresults
3.2Parametersetting of 10-fold cross-validation for all methods are depicted by ROC
|     |     |     |     |     |     |     | curve, PR | curve and top-ranked | results | in  | Figure 2. | As shown in |
| --- | --- | --- | --- | --- | --- | --- | --------- | -------------------- | ------- | --- | --------- | ----------- |
InBNNRalgorithm,therearetwoparametersneededtobedeter-
|     |     |     |     |     |     |     | Figure 2, | the BNNR method | outperforms |     | the other | methods in |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------------- | ----------- | --- | --------- | ---------- |
mined,includingaandb.Fortheparametersaandb,weperform
termsofAUCvaluesoftheROCcurves,precisionsandtop-ranked
| cross-validation | on  | the training | dataset | to determine, |     | which are |     |     |     |     |     |     |
| ---------------- | --- | ------------ | ------- | ------------- | --- | --------- | --- | --- | --- | --- | --- | --- |
indications.Specifically,BNNRreportsAUCvalueof0.932,while
determinedfrom{0.1,1,10,100}.Table1reportsAUCvaluescal-
culatedbyBNNRwhenaandbarerangingfrom{0.1,1,10,100} HGBI,DrugNet,MBiRWandDRRShave0.829,0.868,0.917and
|     |     |     |     |     |     |     | 0.930, respectively. | The | more significant |     | gains are | in precision. |
| --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | ---------------- | --- | --------- | ------------- |
in10-foldcross-validation,wherethebestAUCvaluesaredisplayed
BNNRobtainspredictionprecisionof0.440,whichissignificantly
| in bold. One | can find | that | BNNR achieves | the | best performance |     |     |     |     |     |     |     |
| ------------ | -------- | ---- | ------------- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
higherthanHGBI(0.130),DrugNet(0.192),MBiRW(0.304)and
whena¼1andb¼10.
DRRS(0.375).ItisimportanttonotethatBNNRcansuccessfully
Meanwhile,weterminatetheBNNRalgorithmwhenthefollow-
|     |     |     |     |     |     |     | rank44.0%truedrug–diseaseassociationsattop1,whichis |     |     |     |     | 13.6 |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | ---- |
ingstoppingcriterionsaresatisfied:
|     |     |     |     |     |     |     | and 6.5% | higher than MBiRW | and | DRRS, | respectively. | One true |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------------- | --- | ----- | ------------- | -------- |
jf (cid:5)f j drug–disease association is treated as a retrieved association when
|     | f   | (cid:6) tol1; | kþ1 k | (cid:6) tol2; |     | (11) |     |     |     |     |     |     |
| --- | --- | ------------- | ----- | ------------- | --- | ---- | --- | --- | --- | --- | --- | --- |
k maxf1;jf jg its predicted rank is higher than the specified top rank threshold.
k
|              |                                                         |     |     |     |     |     | These approaches | identify | different | numbers | of true | drug–disease |
| ------------ | ------------------------------------------------------- | --- | --- | --- | --- | --- | ---------------- | -------- | --------- | ------- | ------- | ------------ |
| where f ¼kXk | þ 1(cid:5) X kkF,tol1andtol2arethegiventolerances,which |     |     |     |     |     |                  |          |           |         |         |              |
k k X kk associations with respect to different rank cutoffs, which are pre-
F
aresetas2(cid:3)10(cid:5)3and10(cid:5)5inBNNRalgorithm,respectively. sentedinFigure2c.Forinstance,amongthe1933truedrug–disease
|     |     |     |     |     |     |     | associations, | 1333 associations | are | identified | at top | 5 by BNNR, |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ----------------- | --- | ---------- | ------ | ---------- |
whileincomparison,only561,738,1044and1251associationsare
3.3Comparewithothermethods
predictedbyHGBI,DrugNet,MBiRWandDRRS,respectively.In
BNNRiscomparedwithfourlatestmethodsfordrugrepositioning:
practice,precisionisamoreimportantmeasureofthedrug–disease
| HGBI (Wang | et al., | 2013), | DrugNet | (Martinez | et  | al., 2015), |     |     |     |     |     |     |
| ---------- | ------- | ------ | ------- | --------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
associationpredictionperformance,becauseamoreprecisepredic-
MBiRW(Luoetal.,2016)andDRRS(Luoetal.,2018).Basedon
tionprovidescorrectindicationforexistingdrugswithhigherprob-
theguilt-by-associationprincipleandtheinterpretationofinforma-
ability,whichcanleadtobudgetandtimereduction.
tionflow,HGBIisdesignedforpredictingdisease-associateddrugs.
| DrugNet   | is based on  | propagation | flow         | algorithm,      | which | can per- |     |     |     |     |     |     |
| --------- | ------------ | ----------- | ------------ | --------------- | ----- | -------- | --- | --- | --- | --- | --- | --- |
| form both | drug–disease | and         | disease–drug | prioritization. |       | MBiRW    |     |     |     |     |     |     |
3.4Predictingindicationsfornewdrugs
| and DRRS | are our | previous | works, MBiRW |     | uses comprehensive |     |     |     |     |     |     |     |
| -------- | ------- | -------- | ------------ | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
ToassessthecapabilityofBNNRinpredictingpotentialindications
similaritymeasuresandBiRWalgorithmtoinferdrug–diseaseasso-
fornewdrugs,wechoosethesedrugswhichhaveonlyoneknown
| ciation. DRRS | constructs | a   | heterogeneous | drug–disease |     | network |     |     |     |     |     |     |
| ------------- | ---------- | --- | ------------- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- |
drug–diseaseassociationtoconductadenovotest.Foreachofthese
andconductspredictionbasedonthematrixcompletionofSVTal-
drugs,theknowndiseaseassociationisremovedinturnasthetest
gorithmtopredictpotentialindicationsfordrugs.
sampleandotherexistingassociationsareusedastrainingsample.
| Although           | DRRS | and BNNR | are based | on          | the same | heteroge- |       |                  |           |              |     |              |
| ------------------ | ---- | -------- | --------- | ----------- | -------- | --------- | ----- | ---------------- | --------- | ------------ | --- | ------------ |
|                    |      |          |           |             |          |           | For a | new drug without | any known | drug–disease |     | association, |
| neous drug–disease |      | network, | BNNR      | can exploit | more     | accuracy  |       |                  |           |              |     |              |
BNNRisabletopredictitsdrug–diseaseassociationsbytakingad-
vantageofthesimilarityinformationofthenoveldruginadjacency
Table1.TheAUCvaluesusingdifferentaandbvaluesin10-fold
matrix.Also,duetothefactthatthereisnodrug–diseaseassociation
cross-validationonthegoldstandarddataset
|     |     |     |     |     |     |     | information | for the novel | drug, the | similarity | information | is more |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------------- | --------- | ---------- | ----------- | ------- |
a\b 0.1 1 10 100 important than the existing drug–disease association information
|     |     |     |     |     |     |     | for the | other drugs, which | should | be given | heavier | weights. |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------------ | ------ | -------- | ------- | -------- |
0.1 0.757 0.785 0.879 0.888 Equivalently,associationmatrixismultipliedbyaweightcoefficient
| 1   | 0.863 |     | 0.921 | 0.933 |     | 0.899 |     |     |     |     |     |     |
| --- | ----- | --- | ----- | ----- | --- | ----- | --- | --- | --- | --- | --- | --- |
0.7inthisstudy.
| 10  | 0.854 |     | 0.921 | 0.926 |     | 0.890 |     |     |     |     |     |     |
| --- | ----- | --- | ----- | ----- | --- | ----- | --- | --- | --- | --- | --- | --- |
AsshowninFigure3forthedenovotest,BNNRachievesAUC
| 100 | 0.862 |     | 0.919 | 0.925 |     | 0.889 |          |                    |          |       |     |           |
| --- | ----- | --- | ----- | ----- | --- | ----- | -------- | ------------------ | -------- | ----- | --- | --------- |
|     |       |     |       |       |     |       | value of | 0.830, while HGBI, | DrugNet, | MBiRW | and | DRRS have |

1
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0 0.2 0.4 0.6 0.8 1 False Positive Rate
drug–diseaseassociationsinthegoldstandarddatasetasthetraining
set andregard the missingdrug–diseasepairs as the candidateset.
After the prediction scores of all candidate pairs are computed by
BNNR,werankthecandidatediseasesbythepredictedscoresfor
eachdrug.
In order to confirm whether the predicted diseases are true or
not,wechooseLevodopa,Doxorubicin,AmantadineandFlecainide
as the representative drugs to validate their potential diseases pre-
dictedby BNNR andthen list the confirmedinformation of top-5
candidatediseasesforthem.Weconfirmthepotentialdiseasesasso-
ciatedwiththegivendrugbyauthoritativepublicdatabases,suchas
DrugBank, CTD (Davis et al., 2013) and KEGG (Kanehisa et al.,
2014).Thepredictedresultsandthesupportingevidencesaresum-
inferior results with 0.746, 0.782, 0.818 and 0.824, respectively.
marizedinTable2.Foreachrepresentativedrug,morethanthree
Fortop-rankedresults,BNNRoutperformsallmethodsattop5,10
new drug–disease associations on top-5 have been reported in the
and50,exceptforbeinginferiortoDRRSattop1.
publicdatabases.ItdemonstratestheeffectivenessofBNNRinpre-
dictingnovelindicationsfordrugsinpracticaluse.
3.5Casestudies Furthermore,BNNRidentifiesothernewindicationsincluding:
In these case studies, we apply BNNR to predict new uses for al- Levodopa for hyperplastic myelinopathy; Doxorubicin for dohle
ready approved drugs in practical applications. In the process of bodies;Amantadineforrestlesslegssyndromeandmalignanthyper-
identifying novel drug–disease associations, we treat all known thermia; Flecainide for nephropathy-hypertension and hyperplastic
etaR
evitisoP
eruT
0.45
0.4
0.35
0.3
0.25
0.2
BNNR
DRRS 0.15 MBiRW
DrugNet 0.1
HGBI
0.05
0
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 Recall
noisicerP
1800
BNNR 1600
DRRS
MBiRW 1400
DrugNet HGBI 1200
1000
800
600
400
200
0 Top1 Top5 Top10 Top50
snoitaicossa
esaesid−gurd
deveirter
fo
rebmun
ehT
(a) (b) (c)
BNNR
DRRS
MBiRW
DrugNet
HGBI
Fig.2.Theperformanceofallmethodsinpredictingdrug–diseaseassociationfor10-foldcross-validation.(a)ROCcurveofpredictionresults.(b)PRcurveofpre-
dictingcandidatediseasesfordrugs.(c)Thenumberofcorrectlyretrieveddrug–diseaseassociationsforvariousrankthresholds
0.35
0.3
0.25
0.2
0.15
0.1
0.05
0
0 0.2 0.4 0.6 0.8 1
Recall
noisicerP
BNNR
DRRS
MBiRW
DrugNet
HGBI
140
120
100
80
60
40
20
0
Top1 Top5 Top10 Top50
sgurd
deveirter
fo
rebmun
ehT
i460 M.Yangetal.
(a) Table 2. The top five candidate diseases for Levodopa,
Doxorubicin,AmantadineandFlecainide
Drugs Topfivecandidatediseases Evidences
(DrugBankIDs) (OMIMIDs)
Levodopa Parkinsondisease(168600) KEGG/DB/CTD
(DB01235) Dementia(125320) DB/CTD
Multiplesclerosis(126200) CTD
Pheochromocytoma(171300) CTD
Hyperplasticmyelinopathy(147530)
Doxorubicin Smallcellcancerofthelung(182280) CTD
(DB00997) Dohlebodies(223350)
Testiculargermcelltumor(273300) CTD
Reticulumcellsarcoma(267730) CTD
Leukemia(109543) KEGG/DB/CTD
Amantadine Parkinsondisease(168600) KEGG/DB/CTD
(DB00915) Dementia(125320) DB/CTD
Restlesslegssyndrome(102300)
Alzheimerdisease(104300) CTD
(b)
Malignanthyperthermia(217150)
BNNR Flecainide Atrialfibrillation(608583) CTD
DRRS (DB01195) Cardiacarrhythmia(115000) DB/CTD
MBiRW Diastolichypertension(608622) CTD
DrugNet Nephropathy-hypertension(161900)
HGBI Hyperplasticmyelinopathy(147530)
Fig.3.Theperformanceofallmethodsinpredictingpotentialdiseasesfor
newdrugs.(a)PRcurveofpredictionresults.(b)Thenumberofretrieved
drugsforvariousrankthresholds
Downloaded
from
https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141
by
guest
on
29
April
2020

| DrugrepositioningbasedonBNNR |     |     |     |     |     |     |     |     |     |     |     |     | i461 |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
myelinopathy.Thesepredictedassociationsarenotyetreportedin 3.6Theeffectsofboundedconstrainandregularization
current literature, but may have a greater likelihood of existing. modelofBNNRonperformance
Therearegreatopportunitiestoresearchandvalidatetheseassocia- In order to evaluate the effectiveness of bounded constraint [0, 1]
tionsformedicalresearchersandpharmaceuticalcompanies.
andregularizationmodel,wecompareBNNRwithtwomodelsin
10-foldcross-validation.ThefirstmodelisBNNRwithoutbounded
constraint[0,1](referredtoasNNR),whiletheotheroneisBNNR
withoutregularizationterm(referredtoasBNN).Specifically,NNR
|     |       |     |     |     |     | BNNR | isdefinedas: |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | ---- | ------------ | --- | --- | --- | --- | --- | --- |
|     | 0.945 |     |     |     |     | BNN  |              |     |     |     |     |     |     |
a
|     |      |     |     |     |     | NNR |     |     | m   | inkXk þ | kPXðXÞ(cid:5)PXðMÞk2 | ;   | (12) |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------------------- | --- | ---- |
|     | 0.94 |     |     |     |     |     |     |     |     | (cid:4) |                      | F   |      |
|     |      |     |     |     |     |     |     |     | X   |         | 2                    |     |      |
0.935
seulav CUA ehT andBNNisdefinedas: Downloaded from https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141 by guest on 29 April 2020
0.93
minkXk
(cid:4)
|     | 0.925 |     |     |     |     |     |     |     |     | X   |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(13)
s:t:PXðXÞ¼PXðMÞ
0.92
|     |     |     |     |     |     |     |     |     |     | 0 (cid:6) | X (cid:6) 1: |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------------ | --- | --- |
0.915
Onecanfindthatincorporatingtheregularizationtermleadsto
0.91
|     |     |     |     |     |     |     | morerobustpredictionresultscomparedto |     |     |     |     | simplyminimizingthe |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | ------------------- | --- |
0.905 nuclear norm, where the noise in similarity measures is tolerated.
0.9 Moreover,constrainingthepredictedassociationvalueswithin[0,1]
1 2 3 4 5 6 7 8 9 10 furtherimprovesthepredictionaccuracy.Thisisshowninthe10-fold
cross-validationresultsillustratedinFigure4.
Fig.4.PerformancecomparisonofBNNR,NNRandBNNin10-foldcross-val-
TofurtherverifytherobustnessofBNNR,weincreasinglyadd
idationintermsofAUCvalues
|     |     |     |     |     |     |     | random    | noises | to    | the drug–drug | and disease–disease |               | similarity |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ----- | ------------- | ------------------- | ------------- | ---------- |
|     |     |     |     |     |     |     | matrices. | The    | noise | entries       | are drawn           | independently | from       |
0.94
|     |     |     |     |     |     |      | Nð0;1=20Þ                                                 |     | and noise | rate | is the proportion | of the | contaminated |
| --- | --- | --- | --- | --- | --- | ---- | --------------------------------------------------------- | --- | --------- | ---- | ----------------- | ------ | ------------ |
|     |     |     |     |     |     | BNNR | entrieswithrespecttoallcomponentsofsimilaritymatrix.Weset |     |           |      |                   |        |              |
|     |     |     |     |     |     | BNN  | thenoiseratein[0,0.3]withanincreasestepsizeof0.06.BNNR    |     |           |      |                   |        |              |
0.92
andBNNarecomparedin10-foldcross-validationintermsofAUC
values.Withoutasurprise,asshowninFigure5,theAUCvaluesde-
0.9
|     |     |     |     |     |     |     | crease | gradually | as  | the noise | rate increases | in both | BNNR and |
| --- | --- | --- | --- | --- | --- | --- | ------ | --------- | --- | --------- | -------------- | ------- | -------- |
seulav CUA ehT
BNN.However,thedecreaseofBNNRismuchslowercomparedto
BNN,indicatingthatBNNRisabletobettertoleratenoisysimilar-
0.88
itycomputations.ThisalsoexplainswhyBNNRleadstobetterpre-
|     |      |     |     |     |     |     | diction                          | accuracy | when | the | nuclear norm | regularization | term is |
| --- | ---- | --- | --- | --- | --- | --- | -------------------------------- | -------- | ---- | --- | ------------ | -------------- | ------- |
|     | 0.86 |     |     |     |     |     | incorporated.                    |          |      |     |              |                |         |
|     | 0.84 |     |     |     |     |     | 3.7Experimentsontheotherdatasets |          |      |     |              |                |         |
InordertoillustratetheadaptabilityofBNNRindifferentdatasets,we
0.82 perform BNNR on the two other datasets including Cdataset and
|     | 0   | 0.05 0.1 | 0.15 | 0.2 | 0.25 | 0.3 |            |     |           |      |                 |           |               |
| --- | --- | -------- | ---- | --- | ---- | --- | ---------- | --- | --------- | ---- | --------------- | --------- | ------------- |
|     |     |          |      |     |      |     | DNdataset, |     | which are | used | in our previous | work (Luo | et al., 2016, |
Noise rate
|      |                |               |          |           |           |       | 2018).    | Cdataset | (Luo         | et al.,  | 2016) contains | 663 drugs | collected in |
| ---- | -------------- | ------------- | -------- | --------- | --------- | ----- | --------- | -------- | ------------ | -------- | -------------- | --------- | ------------ |
|      |                |               |          |           |           |       | DrugBank, |          | 409 diseases | obtained | in OMIM        | database  | and 2352     |
| Fig. | 5. Performance | comparison of | BNNR and | BNN under | different | noise |           |          |              |          |                |           |              |
ratesintermsofAUCvalues known drug–disease associations. DNdataset (Martinez et al., 2015)
| (a) |     |     |     | (b)   |     |     |     |     |     | (c)                                                    |      |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | ------------------------------------------------------ | ---- | --- | --- |
|     |     |     |     |       |     |     |     |     |     | snoitaicossa esaesid−gurd deveirter fo rebmun ehT 2500 |      |     |     |
|     | 1   |     |     |   0.5 |     |     |     |     |     |                                                        | BNNR |     |     |
DRRS
|                    | 0.9 |     |     | 0.45          |     |     |     | BNNR    |     |      |         |     |     |
| ------------------ | --- | --- | --- | ------------- | --- | --- | --- | ------- | --- | ---- | ------- | --- | --- |
|                    |     |     |     |               |     |     |     | DRRS    |     | 2000 | MBiRW   |     |     |
|                    | 0.8 |     |     | 0.4           |     |     |     | MBiRW   |     |      | DrugNet |     |     |
|                    |     |     |     |               |     |     |     | DrugNet |     |      | HGBI    |     |     |
| etaR evitisoP eruT |     |     |     | 0.35          |     |     |     |         |     |      |         |     |     |
|                    | 0.7 |     |     |               |     |     |     | HGBI    |     |      |         |     |     |
|                    |     |     |     | noisicerP 0.3 |     |     |     |         |     | 1500 |         |     |     |
0.6
0.25
0.5
|     |     |     | BNNR | 0.2 |     |     |     |     |     | 1000 |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- |
DRRS
|     | 0.4 |     |     | 0.15 |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MBiRW
|     | 0.3 |     | DrugNet | 0.1 |     |     |     |     |     | 500 |     |     |     |
| --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
HGBI
|     | 0.2   |     |     | 0.05 |     |     |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 0.1   |     |     |      | 0   |     |     |     |     | 0   |     |     |     |
0 0.2 0.4 0.6 0.8 1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1   Top1 Top5 Top10 Top50
|     |     | False Positive Rate |     |     |     |     | Recall |     |     |     |     |     |     |
| --- | --- | ------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
Fig.6.Theperformanceofallmethodsinpredictingdrug–diseaseassociationsfor10-foldcross-validationonCdataset.(a)ROCcurveofpredictionresults.(b)PR
curveofpredictingcandidatediseasesfordrugs.(c)Thenumberofcorrectlyretrieveddrug–diseaseassociationsforvariousrankthreshold

| i462 |     |     |     |     |     |     |     |     |     |     | M.Yangetal. |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- |
includes 1490 drugs registered in DrugBank, 4516 diseases anno- ForCdataset,asshowninFigure6,BNNRobtainsAUCvalue
tated by Disease Ontology (DO) terms and 1008 known drug– of 0.948 in 10-fold cross-validation, while HGBI, DrugNet,
diseaseassociations.Weevaluatetherobustnessofourmethodon MBiRWandDRRShave0.858,0.903,0.933and0.947,respective-
thesetwodatasetsbyperforming10-foldcross-validationandthede ly. The PR curves illustrate that BNNR obtains the best precision
novo test. The parameters of BNNR for Cdataset and DNdataset with0.471,whileHGBI,DrugNet,MBiRWandDRRShave0.168,
are set as Section 3.2. (For Cdataset, a¼1 and b¼10. For 0.239, 0.351 and 0.403, respectively. Meanwhile, BNNR outper-
DNdataset,a¼1andb¼1.) formsthe othermethodson toprankresults.Morespecifically, at
top-5rank,1855associationsoutof2532areidentifiedbyBNNR,
whileonly796,1193,1481and1753associationsarepredictedby
|     |     |     |     | HGBI, | DrugNet, | MBiRW | and | DRRS, | respectively. |     | In the de novo |     |
| --- | --- | --- | --- | ----- | -------- | ----- | --- | ----- | ------------- | --- | -------------- | --- |
(a) test,PRcurveandtoprankresultsareillustratedinFigure7.BNNR
| 0.35 |     |     |     |         |      |               |        |             |          |               |           |                                                                                                                       |
| ---- | --- | --- | --- | ------- | ---- | ------------- | ------ | ----------- | -------- | ------------- | --------- | --------------------------------------------------------------------------------------------------------------------- |
|      |     |     |     | obtains | AUC  | value of      | 0.812, | while HGBI, | DrugNet, |               | MBiRW and | Downloaded from https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141 by guest on 29 April 2020 |
|      |     |     |     | DRRS    | have | 0.732, 0.785, | 0.804  | and         | 0.819,   | respectively. | DRRS      |                                                                                                                       |
BNNR
0.3 achieves slightly better performance than BNNR. In addition,
DRRS
MBiRW
BNNRoutperformstheothermethodswithrespecttodifferenttop-
| 0.25 |     |     | DrugNet |        |             |               |     |         |      |               |      |     |
| ---- | --- | --- | ------- | ------ | ----------- | ------------- | --- | ------- | ---- | ------------- | ---- | --- |
|      |     |     |         | ranked | thresholds. | Specifically, |     | for 177 | drug | associations, | BNNR |     |
HGBI
|     |     |     |     | retrieves | 87(49.2%) | drugs | at  | top 10 | rank, while | HGBI, | DrugNet, |     |
| --- | --- | --- | --- | --------- | --------- | ----- | --- | ------ | ----------- | ----- | -------- | --- |
noisicerP
| 0.2 |     |     |     | MBiRW | and | DRRS have | 48(27.1%), |     | 61(34.5%), | 80(45.2%) | and |     |
| --- | --- | --- | --- | ----- | --- | --------- | ---------- | --- | ---------- | --------- | --- | --- |
78(44.0%),respectively.
0.15
ForDNdataset,asshowninFigure8,BNNRobtainsAUCvalue
|     |     |     |     | of  | 0.955 in | 10-fold | cross-validation, |     | while | HGBI, | DrugNet, |     |
| --- | --- | --- | --- | --- | -------- | ------- | ----------------- | --- | ----- | ----- | -------- | --- |
0.1
MBiRWandDRRShave0.921,0.950,0.956and0.934,respective-
ly.ThePRcurvesshowthatBNNRobtainsthebestprecisionwith
| 0.05 |     |     |     | 0.347, | while | HGBI,      | DrugNet,      | MBiRW | and     | DRRS       | have 0.204, |     |
| ---- | --- | --- | --- | ------ | ----- | ---------- | ------------- | ----- | ------- | ---------- | ----------- | --- |
|      |     |     |     | 0.150, | 0.321 | and 0.346, | respectively. |       | It is a | noteworthy | fact that   |     |
0
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 BNNRhasbetterAUCvalueandprecisioncomparedtoothermeth-
|     |     | Recall |     | ods.Meanwhile,BNNRoutperformstheothermethodsontoprank |     |     |     |     |     |     |     |     |
| --- | --- | ------ | --- | ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
(b)
resultsfromfourdifferentthresholds.Indenovotest,PRcurveand
| 120 |     |     |     |         |             |     |             |     |               |          |      |     |
| --- | --- | --- | --- | ------- | ----------- | --- | ----------- | --- | ------------- | -------- | ---- | --- |
|     |     |     |     | toprank | resultsofde |     | novotestare |     | illustratedin | Figure9. | BNNR |     |
BNNR
|     | DRRS |     |     | obtainsAUCvalueof0.956,whichisslightlyworsethanDrugNet |     |     |     |     |     |     |     |     |
| --- | ---- | --- | --- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
100
sgurd deveirter fo rebmun ehT MBiRW andMBiRW,whileHGBIandDRRShave0.928and0.946,respect-
DrugNet
|     |     |     |     | ively. | BNNR | surpasses | the other | methods | on  | top rank | results: for |     |
| --- | --- | --- | --- | ------ | ---- | --------- | --------- | ------- | --- | -------- | ------------ | --- |
HGBI
| 80  |     |     |     | 347testdrugassociations,BNNRretrieves145drugsattop1rank, |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
whileHGBI,DrugNet,MBiRWandDRRShave111,84,136and
| 60  |     |     |     | 134,respectively. |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
40
3.8Computationtimecomparisons
Inordertocomparethecomputationalefficiencyofdifferentmeth-
20
ods,wehaveconducteda10-foldcross-validationonthegoldstand-
|     |           |     |             | ard                                                | dataset, | Cdataset | and DNdataset. |     | The running |     | times of these |     |
| --- | --------- | --- | ----------- | -------------------------------------------------- | -------- | -------- | -------------- | --- | ----------- | --- | -------------- | --- |
| 0   |           |     |             | methodswereobtainedonaLinuxserverwithCPU2.30GHzand |          |          |                |     |             |     |                |     |
|     | Top1 Top5 |     | Top10 Top50 |                                                    |          |          |                |     |             |     |                |     |
128GBmemory,whichareshowninSupplementaryTableS1.The
Fig.7.Theperformanceofallmethodsinpredictingpotentialdiseasesfor average running time of BNNR is more than HGBI and DrugNet
newdrugsonCdataset.(a)PRcurveofpredictionresults.(b)Thenumberof but less than MBiRW and DRRS on the gold standard dataset.
retrieveddrugsforvariousrankthresholds AlthoughHGBIismuchfasterthantheothers,ityieldsthelowest
| (a) |     |     | (b) |     |     | (c) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1   0.35   snoitaicossa esaesid−gurd deveirter fo rebmun ehT 800
BNNR
| 0.9                |     |     |           |     | BNNR    | 700 | DRRS    |     |     |     |     |     |
| ------------------ | --- | --- | --------- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- |
|                    |     |     | 0.3       |     | DRRS    |     | MBiRW   |     |     |     |     |     |
| 0.8                |     |     |           |     | MBiRW   |     |         |     |     |     |     |     |
|                    |     |     |           |     |         | 600 | DrugNet |     |     |     |     |     |
| etaR evitisoP eruT |     |     | 0.25      |     | DrugNet |     | HGBI    |     |     |     |     |     |
| 0.7                |     |     |           |     | HGBI    |     |         |     |     |     |     |     |
|                    |     |     | noisicerP |     |         | 500 |         |     |     |     |     |     |
| 0.6                |     |     | 0.2       |     |         |     |         |     |     |     |     |     |
400
0.5
0.15
|     |     | BNNR    |     |     |     | 300 |     |     |     |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 |     | DRRS    |     |     |     |     |     |     |     |     |     |     |
|     |     | MBiRW   | 0.1 |     |     |     |     |     |     |     |     |     |
| 0.3 |     | DrugNet |     |     |     | 200 |     |     |     |     |     |     |
HGBI
| 0.2 |     |     | 0.05 |     |     | 100 |     |     |     |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0
| 0.1 0   0.2 | 0.4 0.6             | 0.8 | 1 0.1 0.2 0.3 | 0.4 0.5 0.6 0.7 | 0.8 0.9 | 1   | 0    |     |      |       |       |     |
| ----------- | ------------------- | --- | ------------- | --------------- | ------- | --- | ---- | --- | ---- | ----- | ----- | --- |
|             | False Positive Rate |     |               | Recall          |         |     | Top1 |     | Top5 | Top10 | Top50 |     |
Fig.8.Theperformanceofallmethodsinpredictingdrug–diseaseassociationfor10-foldcross-validationonDNdataset.(a)ROCcurveofpredictionresults.(b)
PRcurveofpredictingcandidatediseasesfordrugs.(c)Thenumberofcorrectlyretrieveddrug–diseaseassociationsforvariousrankthreshold

References
0.45
Ada,H.etal.(2002)OnlineMendelianInheritanceinMan(OMIM),aknowl-
0.4 edgebase of human genes andgenetic disorders. Nucleic Acids Res., 30,
52–55.
0.35 Boyd,S.etal.(2011)DistributedOptimizationandStatisticalLearningviathe
AlternatingDirectionMethodofMultipliers.Foundationsand Trendsin
0.3
MachineLearning,3,1–122.
0.25 Cai,J.etal.(2010)Asingularvaluethresholdingalgorithmformatrixcomple-
tion.SIAMJ.Optim.,20,1956–1982.
0.2 Candes,E.etal.(2013)Simpleboundsforrecoveringlow-complexitymodels.
Math.Program.,141,577–589.
0.15
Chen,C.etal.(2012)Matrixcompletionviaanalternatingdirectionmethod.
0.1 IMAJ.Numer.Anal.,32,227–245.
Chong,C.etal.(2007)Newusesforolddrugs.Nature,448,645–646.
0.05 Dai,W.etal.(2015)Matrixfactorization-basedpredictionofnoveldrugindi-
cationsbyintegratinggenomicspace.Comput.Math.MethodsMed.,2015,
0
0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 275045.
Recall Davis,A. et al. (2013) The comparative toxicogenomics database: update
2013.NucleicAcidsRes.,41,D1104–D1114.
Davis,J. et al. (2006) The relationship between precision–recall and ROC
curves. In: ICML ’06: Proceedings of the International Conference on
MachineLearning,NewYork,NY,USA,pp.233–240.
Gottlieb,A.etal.(2011)PREDICT:amethodforinferringnoveldrugindica-
tionswithapplicationtopersonalizedmedicine.Mol.Syst.Biol.,7,496.
Hu,Y.etal.(2013)Fastandaccuratematrixcompletionviatruncatednuclear
norm regularization. IEEE Trans. Pattern Anal. Mach. Intell., 35,
2117–2130.
Kanehisa,M.etal.(2014)Data,information,knowledgeandprinciple:back
tometabolisminKEGG.NucleicAcidsRes.,42,199–205.
Li,Y.andYu,W.(2017)Afastimplementationofsingularvaluethresholding
algorithmusingrecyclingrankrevealingrandomizedsingularvaluedecom-
position.arXiv,1704.05528.
Luo,H.etal.(2016)Drugrepositioningbasedoncomprehensivesimilarity
measuresandBi-randomwalkalgorithm.Bioinformatics,32,2664–2671.
Luo,H.etal.(2018)Computationaldrugrepositioningusinglow-rankmatrixap-
proximationandrandomizedalgorithms.Bioinformatics,34,1904–1912.
Ma,S.etal.(2011)FixedpointandBregmaniterativemethodsformatrixrank
minimization.Math.Program.,128,321–353.
Martinez,V.etal.(2015)DrugNet:network-baseddrug–diseaseprioritization
byintegratingheterogeneousdata.Artif.Intell.Med.,63,41–49.
Paul,S.etal.(2010)HowtoimproveR&Dproductivity:thepharmaceutical
precision and AUC values. Moreover, compared to DrugNet on a
industry’sgrandchallenge.Nat.Rev.DrugDiscov.,9,203–214.
biggerdatasetsuchasDNdataset,BNNRismorecomputationally Ramlatchan,A.etal.(2018)Asurveyofmatrixcompletionmethodsforrec-
efficient. ommendationsystems.BigDataMin.Anal.,1,308–323.
Steinbeck,C. et al. (2003) The Chemistry Development Kit (CDK): an
open-source java library for chemo-and bioinformatics. J. Chem. Inf.
4Conclusions Comput.Sci,34,493–500.
Tanimoto,T.T. (1958) An elementary mathematical theory of classification
ThisstudyhasdevelopedanovelmethodnamedBNNRfordrugrepo-
andprediction.Tech.Rep.,IBMCorp.
sitioning.BNNRnotonlycanrestrictallpredictedmatrixentryvalues
Toh,K.etal.(2010)Anacceleratedproximalgradientalgorithmfornuclear
withinaspecificinterval,butalsoexhibitrobustnesstotoleratepoten- normregularizedleastsquaresproblems.Pac.J.Optim.,6,615–640.
tiallynoisysimilaritycalculations.Theresultsofcross-validationand VanDriel,M.A.etal.(2006)Atext-mininganalysisofthehumanphenome.
denovoexperimentshavedemonstratedthatBNNRisaneffectivepre- Eur.J.Hum.Genet.,14,535–542.
dictionapproach.Especially,comparingwiththeexistingdrugreposi- Wang,W.etal.(2013)Drugtargetpredictionsbasedonheterogeneousgraph
tioningmethods,BNNRyieldsboththebestAUCvalueandthebest inference.Pac.Symp.Biocomput.,18,53–64.
precisioninmostmeasures.Ourcasestudieshaveconfirmedthereli- Wang,W.etal.(2014)Drugrepositioningbyintegratingtargetinformation
throughaheterogeneousnetworkmodel.Bioinformatics,30,2923–2930.
abilityoftheidentifiednewdrug–diseaseassociations.Inthefuture,we
Weininger,D.(1988)SMILES,achemicallanguageandinformationsystem.
plan to integrate drug–target information into the existing heteroge-
1.Introductiontomethodologyandencodingrules.J.Chem.Inf.Comput.
neousnetworkstofurtherimprovethepredictionabilityofBNNR.
Sci.,28,31–36.
Wen,Z.etal.(2010)AlternatingdirectionaugmentedLagrangianmethodsfor
semi-definiteprogramming.Math.Program.Comput.,2,203–230.
Funding
Wishart,D. et al. (2006) DrugBank: a comprehensive resource for in silico
drugdiscoveryandexploration.NucleicAcidsRes.,668–672.
ThisworkwassupportedbytheNationalNaturalScienceFoundationofChina
Yang,J.andYuan,X.(2012)LinearizedaugmentedLagrangianandalternat-
underGrantnumber(61732009,61622213,61772552and61420106009).
ingdirectionmethodsfornuclearnormminimization.Math.Comput.,82,
ConflictofInterest:nonedeclared. 301–329.
noisicerP
BNNR
DRRS
MBiRW
DrugNet
HGBI
300
250
200
150
100
50
0
Top1 Top5 Top10 Top50
sgurd
deveirter
fo
rebmun
ehT
DrugrepositioningbasedonBNNR i463
(a)
(b)
BNNR
DRRS
MBiRW
DrugNet
HGBI
Fig.9.Theperformanceofallmethodsinpredictingpotentialdiseasesfor
newdrugsonDNdataset.(a)PRcurveofpredictionresults.(b)Thenumberof
retrieveddrugsforvariousrankthresholds
Downloaded
from
https://academic.oup.com/bioinformatics/article-abstract/35/14/i455/5529141
by
guest
on
29
April
2020