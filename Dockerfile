# 1. השתמש בתמונת בסיס רשמית של פייתון
FROM python:3.11-slim

# 2. הגדר את תיקיית העבודה בתוך הקונטיינר
WORKDIR /app

# 3. העתק את קובץ התלויות והתקן אותן
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. העתק את שאר קוד האפליקציה
COPY . .

# 5. הגדר את פקודת ההרצה שתפעיל את השרת
# משתנה הסביבה PORT יסופק אוטומטית על ידי Google Cloud
CMD ["functions-framework", "--target=webhook", "--source=main.py", "--port=8080"]