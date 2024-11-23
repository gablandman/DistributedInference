from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from typing import List, Dict
from sqlalchemy import create_engine, Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
import asyncio
import json

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DeviceDB(Base):
    __tablename__ = "devices"
    id = Column(String, primary_key=True, index=True)
    ip = Column(String, index=True)
    name = Column(String)
    available = Column(Boolean, default=True)

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, device_id: str):
        await websocket.accept()
        self.active_connections[device_id] = websocket

    def disconnect(self, device_id: str):
        if device_id in self.active_connections:
            del self.active_connections[device_id]

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, exclude_device_id: str = None, db: Session = Depends(get_db)):
        for device_id, connection in self.active_connections.items():
            if device_id != exclude_device_id:
                db_device = db.query(DeviceDB).filter(DeviceDB.id == device_id).first()
                if db_device and db_device.ip:
                    await connection.send_text(message)

manager = ConnectionManager()

device_vm_association = {
    "1": {"vm_ip": "62.210.195.62", "vm_port": 8000},
    "2": {"vm_ip": "192.168.1.102", "vm_port": 8000},
    "3": {"vm_ip": "192.168.1.103", "vm_port": 8000}
}


def format_curl_message(vm_ip: str, vm_port: int, model: str, conv: List[str]) -> str:
    messages = [{"role": "assistant" if i % 2 == 0 else "user", "content": message} for i, message in enumerate(conv)]
    return json.dumps({
        "action": "execute_curl",
        "url": f"http://{vm_ip}:{vm_port}/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json"
        },
        "data": {
            "model": model,
            "messages": messages,
            "temperature": 0.7
        }
    })

@app.post("/device/open")
async def open_device(request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if db_device:
        if db_device.available:
            raise HTTPException(status_code=400, detail="Device already exists and is available")
        else:
            db_device.available = True
            db.commit()
            db.refresh(db_device)
            return {"message": "Device reactivated successfully", "device_id": db_device.id}
    new_device = DeviceDB(id=str(uuid.uuid4()), ip=ip, name="Device")
    db.add(new_device)
    db.commit()
    db.refresh(new_device)
    return {"message": "Device added successfully", "device_id": new_device.id}

@app.post("/device/close")
async def close_device(request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    db_device.available = False
    db.commit()
    db.refresh(db_device)
    return {"message": "Device marked as unavailable"}

@app.post("/device/request")
async def handle_request(conv: List[str], model: str, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    if not db_device.available:
        raise HTTPException(status_code=400, detail="Device is not available")

    await manager.broadcast("work_start", exclude_device_id=db_device.id, db=db)

    vm_info = device_vm_association.get(db_device.id)
    print(db_device.id)
    print(vm_info)
    if vm_info:
        vm_ip = vm_info["vm_ip"]
        vm_port = vm_info["vm_port"]
        message = format_curl_message(vm_ip, vm_port, model, conv)
        if db_device.id in manager.active_connections:
            await manager.send_message(message, manager.active_connections[db_device.id])

    response = await manager.active_connections[db_device.id].receive_text()

    await manager.send_message(response, manager.active_connections[db_device.id])

    await manager.broadcast("work_over", exclude_device_id=db_device.id, db=db)    

    return {"message": "Request handled successfully"}

@app.get("/devices")
async def read_devices(db: Session = Depends(get_db)):
    devices = db.query(DeviceDB).all()
    return devices

@app.get("/device/id")
async def get_device_id(request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    db_device = db.query(DeviceDB).filter(DeviceDB.ip == ip).first()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"device_id": db_device.id}

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await manager.connect(websocket, device_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_message(f"Message from {device_id}: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(device_id)