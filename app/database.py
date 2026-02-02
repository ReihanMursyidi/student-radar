from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./students.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class StudentDB(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    nisn = Column(String, unique=True, index=True)
    nama = Column(String, index=True)
    pekerjaan_ortu = Column(String)
    anak_ke = Column(Integer)
    jml_saudara = Column(Integer)
    uang_saku = Column(Integer)
    organisasi = Column(String)
    hobi = Column(String)
    kehadiran = Column(Integer)
    nilai = Column(Integer)
    pelanggaran = Column(Integer)
    catatan = Column(Text)

    risk_score = Column(Float)