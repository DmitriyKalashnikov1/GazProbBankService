cd ./AnalyzeService && \
    uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
cd ./DashboardService && \
     streamlit run app.py &
