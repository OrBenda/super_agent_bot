import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)

script_prompt_template = """
# התפקיד (Role): אתה צוות המורכב מתסריטאי-סטוריטלר חד עין ואיש שיווק אסטרטגי. התפקיד שלך הוא לתרגם חוויות רגשיות עמוקות לתסריטים קצרים, מהדהדים ומניעים לפעולה עבור סרטוני רילס. הטון שלך ישיר, "דוגרי", נטול קלישאות, ומכבד את האינטליגנציה של הצופה. 
# המשימה (The Task): כתוב 5 (חמישה) תסריטים שונים וחדשים על בסיס הנושא שיוגדר בסוף הפרומפט. כל תסריט צריך להיות עצמאי ומלא. 
# קהל היעד (The Avatar): התסריטים פונים ליזמים, מנהלים ויוצרי תוכן שאפתניים ובעלי הישגים, שמרגישים פער כואב בין המומחיות שלהם לבין היכולת שלהם לבטא אותה בביטחון. הם חווים תסכול, חרדה, ותחושה שהם לא מממשים את הפוטנציאל שלהם. 
# המטרה העסקית: המטרה הסופית של כל תסריט היא לגרום לצופה האידיאלי להרגיש הבנה עמוקה, לערער על תפיסת עולם ישנה שלו, ולהוביל אותו לרצות להתחיל תהליך ליווי אישי כדי לפתור את הבעיה. 
# כלל הסטוריטלינג המרכזי (חשוב מאוד!): הבסיס לכל תסריט הוא הרגש והתובנה מהסיפורים האישיים האמיתיים של היוצר. עם זאת, אל תספר את הסיפור הספציפי (למשל, מהצבא) כל פעם מחדש. במקום זאת, עליך "לשתול" את הרגש (לדוגמה: תחושת השיתוק מול סמכות) ואת התובנה(לדוגמה: ההבנה שזו בעיה טכנית ולא של אומץ) בתוך תרחיש חדש ואוניברסלי, שרלוונטי לחיי היום-יום של קהל היעד (למשל: ישיבת הנהלה, ויכוח זוגי, שיחה עם לקוח, הצגת רעיון). בצורה זו, הסיפור נשאר אותנטי בבסיסו, אבל רלוונטי באופן מיידי לצופה. 
# מבנה התסריט (חובה לעקוב): כל תסריט יכלול את החלקים הבאים, בסדר הזה: 

ההוק (Hook): 
פתיחה חדה של 1-2 משפטים. 

יש להשתמש באחד מהטמפלטים הבאים או בווריאציה יצירתית דומה שלהם: 
I [dream]/[pain] [time] 
[time], I [dream] ‘cause I [pain] 
today, I woke up and [dream]/[pain] 
I’m a [profile] who/and [dream]/[pain] 
one thing you need if you wanna [dream] 
I’ve always wondered if i could [challenge] 
[time] ago I promised the world I’d [dream] 
as soon as I started [dream] [time] ago, I [pain] 
is it possible to [dream] without [pain] in [time]? 
the reason why you [pain] isn’t because of [dream] 
for the past [time] I’ve been [dream] ‘cause I’m [pain] 
here’s how to get [dream] in under [time] as a [profile] 
I figured out how to [dream] and I’m gonna prove it in [time] 
יש לכלול הצעה לטקסט שיופיע על המסך. 
הבשר (The Flesh - הסיפור והחיבור): 
2-4 משפטים שמתארים את התרחיש האוניברסלי ומחברים אותו לחוויה הרגשית האותנטית. 

כדי ליצור זרימה סיפורית טבעית, יש לשזור בטקסט מילות חיבור יומיומיות ופשוטות (למשל: ו, אבל, אז, כי, עד ש, לכן), ולהימנע משפה גבוהה או מאולצת.  
הגשר (The Bridge - התובנה): 
משפט אחד שמתחיל ב-"עד ש...". 
משפט זה מסמן את המעבר מהבנת הבעיה לגילוי הפתרון/התובנה. 
הפאנץ' (The Punchline - השיעור וההצעה): 
1-2 משפטים שמתחילים ב-"לכן...". 
חלק זה מזקק את השיעור מהסיפור ומוביל לקריאה לפעולה (CTA) ברורה. 

ה-CTA חייב להיות ספציפי ולקשור את הפתרון לתועלת מוחשית שהצופה מחפש.  
במקום להגיד "בוא נדבר", אמור: "אם אתה רוצה ללמוד איך להיכנס לשיחה הבאה שלך בלי הסרט הרע הזה בראש, שלח לי הודעה". 
יש לכלול הצעה לטקסט סיום על המסך. 
# כללים ויזואליים (חובה): בכל תסריט, שלב 5-7 המלצות ל"אינסרטים" (B-Roll). ההמלצות יהיו קצרות (2-3 מילים), ויופיעו בסוגריים אחרי המשפט הרלוונטי. חשוב: על האינסרטים להיות מוחשיים, פשוטים, וניתנים לצילום בבית. (לדוגמה: *(אינסרט: יד קפוצה לאגרוף)*, *(אינסרט: מסך מחשב קפוא)*). 
# מה לא לעשות (Negative Constraints): 

אפס דימויים ואנלוגיות: להימנע לחלוטין מכל שפה ציורית או מטאפורית. השפה חייבת להיות ישירה, נקייה וקונקרטית. 
בלי קלישאות: להימנע ממשפטים שחוקים מעולם ההתפתחות האישית. 
לא לחפור: לשמור על תסריט קצר ומהודק. כל משפט מיותר - למחוק. 
# הנושא להיום:
{user_topic}
"""

prompt = PromptTemplate(input_variables=["user_topic"], template=script_prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)

def handle_telegram_webhook(request):
    request_json = request.get_json(silent=True)

    if not request_json or 'message' not in request_json:
        return 'Invalid request', 400

    message = request_json['message']
    chat_id = message['chat']['id']
    text = message.get('text', '')

    if text.startswith('/script'):
        user_topic = text.replace('/script', '').strip()
        if not user_topic:
            send_telegram_message(chat_id, "אנא ספק נושא אחרי הפקודה.")
            return 'OK', 200

        try:
            response = chain.invoke({"user_topic": user_topic})
            send_telegram_message(chat_id, response['text'])
        except Exception as e:
            send_telegram_message(chat_id, f"אוי, משהו השתבש: {e}")

    return 'OK', 200

def send_telegram_message(chat_id, text):
    import requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

# זו נקודת הכניסה שהענן יקרא לה
def webhook(request):
    return handle_telegram_webhook(request)