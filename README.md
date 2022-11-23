# Graph Component Contrastive Learning
## Abstract
Concept relatedness estimation (CRE) aims to determine whether two given concepts are related. Existing methods only consider the pairwise relationship between concepts, while overlooking the higher-order relationship that could be encoded in a concept-level graph structure. We discover that this underlying graph satisfies a set of intrinsic properties of CRE, including reflexivity, commutativity, and transitivity. In this paper, we formalize the CRE properties and introduce a graph structure named ConcreteGraph. To address the data scarcity issue in CRE, we introduce a novel data augmentation approach to sample new concept pairs from the graph. As it is intractable for data augmentation to fully capture the structural information of the ConcreteGraph due to a large amount of potential concept pairs, we further introduce a novel Graph Component Contrastive Learning framework to implicitly learn the complete structure of the ConcreteGraph. Empirical results on three datasets show significant improvement over the state-of-the-art model. Detailed ablation studies demonstrate that our proposed approach can effectively capture the high-order relationship among concepts.

## Our model
<object data="h./images/overview.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="./images/overview.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="./images/overview.pdf">Download PDF</a>.</p>
    </embed>
</object>
<!-- <embed src="./images/overview.pdf" width="350"> -->
