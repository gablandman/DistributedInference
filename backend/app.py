from fastapi import FastAPI, HTTPException, Depends, Request
from typing import List
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DeviceDB(Base):
    __tablename__ = "devices"
    ip = Column(String, primary_key=True, index=True)
    name = Column(String)

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/device/open")
async def open_device(request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if db_device:
        raise HTTPException(status_code=400, detail="Device already exists")
    new_device = DeviceDB(ip=ip, name="Device")
    db.add(new_device)
    db.commit()
    db.refresh(new_device)
    return {"message": "Device added successfully"}

@app.delete("/device/close")
async def close_device(request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    db.delete(db_device)
    db.commit()
    return {"message": "Device removed successfully"}

@app.post("/device/request")
async def handle_request(conv: List[str], request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"message": "Request handled successfully"}

@app.get("/devices")
async def read_devices(db: Session = Depends(get_db)):
    devices = db.query(DeviceDB).all()
    return devices

