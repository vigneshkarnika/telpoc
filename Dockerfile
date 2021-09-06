FROM vigneshforegoing/facedet:latest
RUN pip install fastapi uvicorn numpy opencv-python-headless deepface aiofiles python-multipart
EXPOSE 8080
COPY ./app /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]