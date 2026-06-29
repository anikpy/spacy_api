from fastapi import FastAPI, Request, HTTPException, UploadFile, File
import spacy
from fastapi.responses import JSONResponse
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from PIL import Image
import io
from functions import ocr_from_pdf, ocr_from_image, ocr_from_word, ocr_from_excel

app = FastAPI()

# Load SpaCy models lazily
nlp_en_core_web_sm = spacy.load("en_core_web_sm")
nlp_de_core_news_sm = spacy.load("de_core_news_sm")
nlp_de_core_news_md = spacy.load("de_core_news_md")


@app.post("/analyze/en_core_web_sm")
async def en_core_web_sm(text: str):
    if not text.strip():
        return {"error": "Input text is empty"}
    try:
        doc = nlp_en_core_web_sm(text)
        stop_words = nlp_en_core_web_sm.Defaults.stop_words
        tokens = [{"text": token.text, "pos": token.pos_, "entity": token.ent_type_,
                   "label": token.ent_iob_} for token in doc]
        return tokens, stop_words
    except Exception as e:
        return {"error": str(e)}


@app.get("/analyze/de_core_news_sm")
async def de_core_news_sm(text: str):
    doc = nlp_de_core_news_sm(text)
    stop_words = nlp_de_core_news_sm.Defaults.stop_words
    tokens = [{"text": token.text, "pos": token.pos_, "entity": token.ent_type_,
               "label": token.ent_iob_} for token in doc]
    return tokens, stop_words


@app.post("/analyze/de_core_news_md")
async def de_core_news_md(text):
    if not text.strip():
        return {"error": "Input text is empty"}
    try:
        doc = nlp_de_core_news_md(text)
        tokens = [{"text": token.text, "pos": token.pos_, "entity": token.ent_type_,
                   "label": token.ent_iob_} for token in doc]
        return tokens
    except Exception as e:
        return {"error": str(e)}


def get_lang_detector(nlp, name):
    return LanguageDetector()


Language.factory("language_detector", func=get_lang_detector)
nlp_en_core_web_sm.add_pipe('language_detector', last=True)


@app.post("/analyze/lang_detector")
async def lang_detector(request: Request):
    lang = await request.json()
    if 'LangText' not in lang or not lang['LangText']:
        raise HTTPException(status_code=400, detail="Please provide LangText")

    doc = nlp_en_core_web_sm(lang['LangText'])
    language = doc._.language
    return {
        "LangCode": language['language'],
        # "confidence": language['score']
    }


@app.post("/perform_ocr/")
async def perform_ocr(file: UploadFile = File(...)) -> JSONResponse:
    try:
        file_type = file.filename.split('.')[-1].lower()
        file_content = await file.read()
        file_io = io.BytesIO(file_content)

        if file_type == 'pdf':
            extracted_text = ocr_from_pdf(file_io)
        elif file_type in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
            image = Image.open(file_io)
            extracted_text = ocr_from_image(image)
        elif file_type == 'docx':
            extracted_text = ocr_from_word(file_io)
        elif file_type == 'xlsx':
            extracted_text = ocr_from_excel(file_io)
        else:
            return JSONResponse(content={
                "error": "Unsupported file format. Please upload a PDF, image, Word document, or Excel spreadsheet."},
                                status_code=400)

        return JSONResponse(content={"extracted_text": extracted_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
