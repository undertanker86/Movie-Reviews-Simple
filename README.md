# How to run:
1. Run main.py to use FastAPI. (Only call API)
2. Run gradio_app to use Gradio app. (Include FE and BE)
3. If want to use Dockerfile:
- Build Image: ```docker build -t imdb-sentiment .```
- Run Container: ```docker run -d --name imdb-sentiment -p 8000:8000 -p 7860:7860 imdb-sentiment```