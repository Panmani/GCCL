# Graph Component Contrastive Learning
Concept relatedness estimation (CRE) aims to determine whether two given concepts are related. Existing methods only consider the pairwise relationship between concepts, while overlooking the higher-order relationship that could be encoded in a concept-level graph structure. We discover that this underlying graph satisfies a set of intrinsic properties of CRE, including reflexivity, commutativity, and transitivity. In this paper, we formalize the CRE properties and introduce a graph structure named ConcreteGraph. To address the data scarcity issue in CRE, we introduce a novel data augmentation approach to sample new concept pairs from the graph. As it is intractable for data augmentation to fully capture the structural information of the ConcreteGraph due to a large amount of potential concept pairs, we further introduce a novel Graph Component Contrastive Learning framework to implicitly learn the complete structure of the ConcreteGraph. Empirical results on three datasets show significant improvement over the state-of-the-art model. Detailed ablation studies demonstrate that our proposed approach can effectively capture the high-order relationship among concepts.

## Our model
<img src="images/overview.png" width="100%">

## ConcreteGraph
<img src="images/graph.png" width="50%">

## Dependencies
```
scikit-learn
pandas
networkx
transformers
...
```
You can install them with:
```
pip install -r requirements.txt
```

## Usage
```
python code/train.py \
    --model=[TRANSFORMER NAME] \
    --config=[PATH TO CONFIG FILE] \
    --split_file=[PATH TO DATA SPLIT JSON FILE] \
    --dataset=[DATASET NAME] \
    --random_seed=42 \
    --batch_size=8 \
    --num_epochs=2 \
    --learning_rate=1e-5 \
    --use_aug \
    --aug_ratio=2 \
    --k_hops=2 \
    --use_gccl \
```