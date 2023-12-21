https://github.com/czodrowskilab/Machine-learning-meets-pKa

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7096188/


**NOTE:** This model was build for monoprotic structures regarding a pH range of 1 to 14. Lipinsky rules were removed., but max MW set yo 750.
If the model is used with multiprotic structures, the predicted values will probably not
be correct.

## Datasets
Combined all datasets from paper, cleaned, removed duplicates (used mean for dupl. pka) and calculated my own RDKit descriptors.
1. `AvLiLuMoVe.sdf` - Manually combined literature p<i>K</i><sub>a</sub> data<sup>[3]</sup>
2. `chembl25.sdf` - Experimental p<i>K</i><sub>a</sub> data extracted from ChEMBL25<sup>[4]</sup>
3. `datawarrior.sdf` - p<i>K</i><sub>a</sub> data shipped with DataWarrior<sup>[5]</sup>
4. `novartis_cleaned_mono_unique_notraindata.sdf` - [Preprocessed](#prep) data from an inhouse
dataset provided by Novartis<sup>[6]</sup>
<br>
#TODO: make my own split for test/train and maybe even external, then do my own RF as previously
maybe add my own descriptors from TDs ?

still not better. what the ...