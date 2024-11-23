from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import asyncio
import json

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

class Device(BaseModel):
    name: str

class RequestModel(BaseModel):
    conv: List[str]
    model_name: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

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
async def handle_request(request_model: RequestModel, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    devices = db.query(DeviceDB).all()
    peers = {}
    for i, device in enumerate(devices, start=1):
        peers[f"node{i}"] = {
            "address": device.ip,
            "port": 50050 + i,
            "device_capabilities": {
                "model": "Unknown Model",
                "chip": "Unknown Chip",
                "memory": 0,
                "flops": {
                    "fp32": 0,
                    "fp16": 0,
                    "int8": 0
                }
            }
        }
    
    config = {
        "peers": peers
    }
    
    return config

@app.get("/devices")
async def read_devices(db: Session = Depends(get_db)):
    devices = db.query(DeviceDB).all()
    return devices

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Envoyer un message pour afficher le widget computing
            print("Envoi : SHOW_COMPUTING")
            await websocket.send_text("SHOW_COMPUTING")
            await asyncio.sleep(5)

            # Envoyer un message pour masquer le widget computing
            print("Envoi : HIDE_COMPUTING")
            await websocket.send_text("HIDE_COMPUTING")
            await asyncio.sleep(5)

            data = await websocket.receive_text()
            await manager.broadcast(f"Message from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client disconnected")

# To run the application, use the command: uvicorn api:app --reload