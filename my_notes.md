https://github.com/czodrowskilab/Machine-learning-meets-pKa

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7096188/

for me, I won't require the pka calcualtion by marvin, just need the datasets and the training portion<br>
use the edataset #5 for training



**NOTE:** This model was build for monoprotic structures regarding a pH range of 2 to 12.
If the model is used with multiprotic structures, the predicted values will probably not
be correct.

## Datasets

1. `AvLiLuMoVe.sdf` - Manually combined literature p<i>K</i><sub>a</sub> data<sup>[3]</sup>
2. `chembl25.sdf` - Experimental p<i>K</i><sub>a</sub> data extracted from ChEMBL25<sup>[4]</sup>
3. `datawarrior.sdf` - p<i>K</i><sub>a</sub> data shipped with DataWarrior<sup>[5]</sup>
4. `combined_training_datasets_unique.sdf` -  [Preprocessed](#prep) and combined data 
from datasets (2) and (3), used as training dataset and prepared with *QUACPAC/Tautomers*<sup>[2]</sup>
5. `combined_training_datasets_unique_no_oe.sdf` -  [Preprocessed](#prep) and combined data 
from datasets (2) and (3), prepared with RDKit MolVS instead of *QUACPAC/Tautomers*<sup>[2]</sup>
6. `AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf` - [Preprocessed](#prep) data from dataset (1),
used as external testset
7. `novartis_cleaned_mono_unique_notraindata.sdf` - [Preprocessed](#prep) data from an inhouse
dataset provided by Novartis<sup>[6]</sup>, used as external testset
