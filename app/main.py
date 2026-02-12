import uvicorn
import sys
import os
from sqlalchemy.orm import Session
from fastapi import FastAPI, Request, Form, Depends, HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.services import predict_risk_rf, analyze_with_gemini
from app.database import SessionLocal, engine, Base, StudentDB

Base.metadata.create_all(bind=engine)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    students = db.query(StudentDB).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "students": students})

@app.post("/add_student", response_class=HTMLResponse)
async def add_student(
    request: Request,
    db: Session = Depends(get_db),
    nisn: str = Form(...),
    nama: str = Form(...),
    pekerjaan_ortu: str = Form(...),
    anak_ke: int = Form(...),
    jml_saudara: int = Form(...),
    uang_saku: int = Form(...), 
    organisasi: str = Form(default="Tidak ada"),
    hobi: str = Form(default="Tidak ada"),
    kehadiran: int = Form(...),
    nilai: int = Form(...),
    pelanggaran: int = Form(...),
    catatan: Optional[str] = Form(default="-"),
):
    if not catatan or catatan.strip() == "":
        catatan = "Tidak ada catatan khusus."
        
    risk_score = predict_risk_rf(
        kehadiran=kehadiran, 
        nilai=nilai, 
        pelanggaran=pelanggaran, 
        uang_saku=uang_saku, 
        saudara=jml_saudara
    )

    new_student = StudentDB(
        nisn=nisn, nama=nama, pekerjaan_ortu=pekerjaan_ortu,
        anak_ke=anak_ke, jml_saudara=jml_saudara, uang_saku=uang_saku,
        organisasi=organisasi, hobi=hobi, kehadiran=kehadiran,
        nilai=nilai, pelanggaran=pelanggaran, catatan=catatan,
        risk_score=round(risk_score * 100)
    )
    db.add(new_student)
    db.commit()

    # Redirect to dashboard
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "students": db.query(StudentDB).all(),
        "message": f"Data {nama} berhasil ditambahkan!"
    })

@app.get("/analyze_detail/{student_id}", response_class=HTMLResponse)
async def analyze_detail(student_id: int, db: Session = Depends(get_db)):
    student = db.query(StudentDB).filter(StudentDB.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    data_lengkap = {
        "nisn": student.nisn,
        "nama": student.nama,
        "anak_ke": student.anak_ke,
        "jml_saudara": student.jml_saudara,
        "pekerjaan_ortu": student.pekerjaan_ortu,
        "uang_saku": f"{student.uang_saku:,}".replace(",", "."),
        "organisasi": student.organisasi,
        "hobi": student.hobi,
        "kehadiran": student.kehadiran,
        "nilai": student.nilai,
        "pelanggaran": student.pelanggaran,
        "risk_score": student.risk_score,
        "catatan": student.catatan
    }
    
    gemini_analysis = analyze_with_gemini(data_lengkap)

    return JSONResponse(content={
        "analysis": gemini_analysis,
        "data": data_lengkap
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)