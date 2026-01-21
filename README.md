# PlanktonShift (TensorFlow + Streamlit)
Cross-instrument plankton image classification + calibration + Grad-CAM + simple trait extraction.

## 1) Data Available
- ZooScanNet (ZooScan ROIs)(https://www.seanoe.org/data/00446/55741/)
- IFCB images (Imaging FlowCytobot)
    --WHOI-Plankton(https://github.com/hsosik/WHOI-Plankton)
    --IFCB Data Dashboard (download ROIs/Zips by dataset)(https://ifcb-data.whoi.edu/)
    --SMHI IFCB Plankton Image Reference Library(SMHI IFCB Plankton Image Reference Library)

### 1.2) Data Used
-For this experiment WHOI-Plankton (IFCB-style) is used as the Data Source
    --WHOI-Plankton(https://github.com/hsosik/WHOI-Plankton) 
    
        --and saved to data/ifcb
    
    --PlanktonSet 1.0 (2015 National Data Science Bowl images)(https://catalog.data.gov/dataset/planktonset-1-0-plankton-imagery-data-collected-from-f-g-walton-smith-in-straits-of-florida-fro/resource/404a63dd-3912-490f-a82a-a7d244bbf47b)    
    
        --and saved to data/zooscan

    
    --class names were mapped to label_map.yaml as seen here https://whoigit.github.io/whoi-plankton/index.html

    NOTE: The original dataset contains over 3.5 million images. To fit compute constraints and accelerate experimentation, model was trained on a reproducible subset (≈75% of the data), selected via stratified sampling to preserve class proportions.

    

## 1.3) Data layout
Place images in:
- data/ifcb/images/<raw_class>/*.png|jpg
- data/zooscan/images/<raw_class>/*.png|jpg

Edit: src/mapping/label_map.yaml to map raw_class -> coarse_class.

or --Auto-generate label_map.yaml from your IFCB folders
    python scripts/make_label_map_from_folders.py


## 2) Install
pip install -r requirements.txt

## 3) Train (source + optional CORAL alignment)
Train IFCB as source, ZooScan as target (unlabeled stream for CORAL):
python -m src.train --source ifcb --target zooscan --use_coral --out models/planktonshift_source

Or baseline:
python -m src.train --source ifcb --target zooscan --out models/planktonshift_source

## 4) Evaluate on each dataset
python -m src.eval --model models/planktonshift_source --dataset ifcb --out reports/artifacts/eval_ifcb.json
python -m src.eval --model models/planktonshift_source --dataset zooscan --out reports/artifacts/eval_zooscan.json

## 5) Calibrate (temperature scaling)
python -m src.calibrate --model models/planktonshift_source --dataset ifcb --out reports/artifacts/cal_ifcb.json
python -m src.calibrate --model models/planktonshift_source --dataset zooscan --out reports/artifacts/cal_zooscan.json

## 6) Run Streamlit demo
Run after training (and optionally after eval+calibrate for tab 3)

streamlit run app/streamlit_app.py

## 7) References
Olson Robert J. , Sosik Heidi M. , (2007), A submersible imaging-in-flow instrument to analyze nano-and microplankton: Imaging FlowCytobot, Limnol. Oceanogr. Methods, 5, doi:10.4319/lom.2007.5.195. Accessed [19/01/2026].

Cowen, Robert K.; Sponaugle, Su; Robinson, Kelly L.; Luo, Jessica; Guigand, Cedric (2015). PlanktonSet 1.0: Plankton imagery data collected from F.G. Walton Smith in Straits of Florida from 2014-06-03 to 2014-06-06 and used in the 2015 National Data Science Bowl (NCEI Accession 0127422). NOAA National Centers for Environmental Information. Dataset. https://doi.org/10.7289/v5d21vjd. Accessed [21/01/2026].