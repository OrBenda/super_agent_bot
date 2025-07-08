import os
import logging
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# הגדרת לוגינג
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# טעינת משתני סביבה
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- הגדרת המוח והפרומפט המשודרג ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)

# הפרומפט האולטימטיבי לפקודת /script
script_prompt_template = """
# התפקיד (Role):
אתה צוות אסטרטגי-קריאייטיבי המורכב מתסריטאי, פסיכולוג ואיש שיווק. תפקידך לתרגם נושאים ורגשות לתסריטים קצרים ומהדהדים עבור סרטוני רילס. הטון שלך ישיר, "דוגרי", ומכבד את האינטליגנציה של הצופה.

# קהל היעד (The Avatar):
התסריטים פונים ליזמים, מנהלים ויוצרי תוכן שאפתניים. הם חווים פער בין המומחיות שלהם לביטחון לבטא אותה.
{audience_focus_instruction}

# המשימה (The Task):
כתוב 4 תסריטים שונים וחדשים על בסיס הנושא.
{wildcard_instruction}

# הדגש הרגשי (Emotional Core):
התסריטים חייבים להיות בנויים סביב הרגש המרכזי הבא: **{emotion}**. על כל תסריט לבטא את הרגש הזה, לחקור אותו, ולהציע דרך החוצה ממנו.

# סגנון מוכח שעובד (Proven Style - Optional):
אם ניתנו למטה דוגמאות לתסריטים קודמים שהצליחו, השתמש בהם כהשראה מרכזית לסגנון ולטון, אך ודא שהתסריטים החדשים יהיו מקוריים לחלוטין.
{feedback_examples}

# מבנה וחוקים (חובה לעקוב):
1.  **מבנה:** כל תסריט יכלול: הוק (Hook), בשר (Flesh), גשר (Bridge - מתחיל ב"עד ש..."), ופאנץ' (Punchline - מתחיל ב"לכן...").
2.  **אינסרטים ויזואליים:** שלב 5-7 אינסרטים מוחשיים (B-Roll) בכל תסריט, בסוגריים. (לדוגמה: (אינסרט: יד קפוצה לאגרוף)).
3.  **ניתוח אסטרטגי:** בסוף כל תסריט, הוסף שורה אחת: `// למה זה עובד: [הסבר פסיכולוגי/שיווקי קצר]`.
4.  **מה לא לעשות:** אפס קלישאות, אפס דימויים ומטאפורות. שפה ישירה ונקייה.

# הנושא והסגנון להיום:
**נושא:** {topic}
**סגנון:** {style}

---
SCRIPTS:
"""

# הפרומפט הייעודי לפקודת /hooks
hook_prompt_template = """
You are a world-class copywriter specializing in viral hooks for short-form video.
Your task is to generate 7 different, sharp hooks for the given topic.
Each hook should use a different psychological trigger (e.g., curiosity, controversy, pain point, a provocative question, a bold statement).

**Topic:** {topic}
---
HOOKS:
1.
2.
3.
4.
5.
6.
7.
"""

# --- פונקציות הבוט של טלגרם ---

def parse_input(text):
    """Parses the user input string into a dictionary."""
    params = {}
    parts = text.split('|')
    for part in parts:
        try:
            key, value = part.split(':', 1)
            params[key.strip().lower()] = value.strip()
        except ValueError:
            continue
    return params

async def script_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /script command logic."""
    if not context.args:
        await update.message.reply_text("אנא ספק פרטים. מבנה לדוגמה:\n/script נושא: ... | סגנון: ... | רגש: ...")
        return

    input_text = ' '.join(context.args)
    params = parse_input(input_text)

    if 'נושא' not in params or 'סגנון' not in params or 'רגש' not in params:
        await update.message.reply_text("חסרים פרמטרים. חובה לציין: נושא, סגנון, ורגש.")
        return
    
    await update.message.reply_text("קיבלתי. מתחיל לחשוב על תסריטים אסטרטגיים... 🤖")

    # Preparing variables for the prompt
    wildcard_instruction = "בתסריט מספר 4, אתה רשאי לשבור כלל אחד באופן יצירתי כדי להפתיע." if params.get('wildcard') == 'true' else ""
    audience_focus = params.get('פוקוס_קהל', 'כללי')
    audience_focus_instruction = f"התאם את השפה והטון לקהל של '{audience_focus}'. אם זה 'מתחילים', השתמש בשפה פשוטה ומעודדת. אם זה 'מומחים', השתמש בשפה מאתגרת וישירה."
    feedback_examples = "# אין דוגמאות קודמות." # In a real app, this would fetch from a database.

    prompt = PromptTemplate(template=script_prompt_template, input_variables=["topic", "style", "emotion", "audience_focus_instruction", "wildcard_instruction", "feedback_examples"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.invoke({
        "topic": params['נושא'],
        "style": params['סגנון'],
        "emotion": params['רגש'],
        "audience_focus_instruction": audience_focus_instruction,
        "wildcard_instruction": wildcard_instruction,
        "feedback_examples": feedback_examples
    })
    
    await update.message.reply_text(response['text'])

async def hooks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /hooks command."""
    topic = ' '.join(context.args)
    if not topic:
        await update.message.reply_text("אנא ספק נושא. למשל: /hooks פחד קהל")
        return
    
    await update.message.reply_text(f"בסדר, מייצר 7 וריאציות של הוקים לנושא: {topic}...")
    
    prompt = PromptTemplate(template=hook_prompt_template, input_variables=["topic"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"topic": topic})
    
    await update.message.reply_text(response['text'])

def main():
    """Starts the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("script", script_command))
    application.add_handler(CommandHandler("hooks", hooks_command))
    
    print("Strategic Partner Bot is now running...")
    application.run_polling()

if __name__ == '__main__':
    main()