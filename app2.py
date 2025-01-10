from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please ensure it is set in the .env file.")

# Initialize Chat Model
chat_model = ChatOpenAI(
    temperature=0.7,
    model="gpt-4o-mini",  # or "gpt-3.5-turbo" depending on your needs
    api_key=openai_api_key
)

# Initialize FastAPI app
app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Mount static files directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Define Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent recipe assistant. Respond to user queries as follows:
- Provide detailed, helpful, and user-friendly responses to human food recipe-related questions.
- Include cooking procedures, ingredients, alternatives, cuisine suggestions, and cultural insights when relevant.
- If a user asks about animal food recipes, politely inform them that you only assist with human food recipes.
- If a user asks about their previous steps (e.g., "What are my previous steps?"), summarize only the steps related to the specific recipe or topic the user mentions (e.g., "pasta"), omitting greetings and unrelated messages.
- If a user asks an unrelated question (e.g., weather or sports), politely inform them that you can only assist with recipes.

Conversation so far:
{history}

User input: {input}

Your response:
""")

# Set up memory for context
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=chat_model,
    prompt=prompt_template,
    memory=memory
)

class UserInput(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(user_input: UserInput):
    try:
        response = recipe_chatbot(user_input.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def recipe_chatbot(user_input: str) -> str:
    try:
        # Handle special query about previous steps
        if "previous steps" in user_input.lower():
            chat_history = memory.chat_memory.messages
            if not chat_history:
                return "There are no previous steps yet."

            recipe_keyword = None
            for word in user_input.lower().split():
                if word not in ["previous", "steps", "the", "about", "is", "what"]:
                    recipe_keyword = word
                    break

            steps = [msg.content for msg in chat_history if recipe_keyword and recipe_keyword in msg.content.lower()]
            if not steps:
                return f"No steps related to '{recipe_keyword}' have been discussed so far."

            return f"Here are your previous steps related to '{recipe_keyword}':\n\n" + "\n".join(steps)

        # Generate chatbot response
        response = conversation.run(input=user_input)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)