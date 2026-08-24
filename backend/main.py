from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import database

app = FastAPI(
    title="Styla API",
    description="Backend API for Styla - AI Personal Stylist",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
database.init_db()

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Styla API is running"}

class UploadResponse(BaseModel):
    id: int
    filename: str
    category: str
    color: str
    pattern: str

@app.post("/api/wardrobe/upload", response_model=UploadResponse)
async def upload_wardrobe_item(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    # Dummy processing:
    # In a real scenario, this is where rembg, attribute detection, and FashionCLIP run.
    # For now, we mock the attribute extraction.
    dummy_category = "T-Shirt"
    dummy_color = "Blue"
    dummy_pattern = "Solid"

    # Save to database
    db_item = database.WardrobeItem(
        filename=file.filename,
        category=dummy_category,
        color=dummy_color,
        pattern=dummy_pattern
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    return UploadResponse(
        id=db_item.id,
        filename=db_item.filename,
        category=db_item.category,
        color=db_item.color,
        pattern=db_item.pattern
    )
