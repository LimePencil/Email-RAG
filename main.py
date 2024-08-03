from fastapi import FastAPI, Depends, APIRouter, Request
#import models
#from database import engine
from routers import chat
from starlette.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


#models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/",response_class=HTMLResponse)
async def render_homepage(request: Request):
    return templates.TemplateResponse('home.html',{"request":request})

@app.get("/test",response_class=HTMLResponse)
async def test(request: Request):
    return templates.TemplateResponse('test.html',{"request":request})


app.include_router(chat.router)
