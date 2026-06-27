# AI-Based-Detection-of-Synthetic-Video-and-Audio-Media


## How to Run

### 1. Create a virtual environment

```bash
python -m venv venv
2. Activate the virtual environment

Windows:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the main Streamlit app
streamlit run app.py
5. Run the live detection app
streamlit run app_yolo_live.py
Notes
The main app supports uploaded media analysis.
The live app is used for real-time webcam detection.
Large datasets and trained model files are not included in this repository due to file size limits.
THE datasets i used is faceforensic++ and asvspoof2019 from kaggle ithink
