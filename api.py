from fastapi import FastAPI, Depends, Query
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager

# Veritabanı Yapılandırması
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:12345@localhost:5432/testDB"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Modelleri
class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    users = relationship("User", back_populates="department")

class Rank(Base):
    __tablename__ = "ranks"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    users = relationship("User", back_populates="rank")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    department_id = Column(Integer, ForeignKey("departments.id"))
    rank_id = Column(Integer, ForeignKey("ranks.id"))
    
    department = relationship("Department", back_populates="users")
    rank = relationship("Rank", back_populates="users")

# Pydantic Şemaları
class DepartmentResponse(BaseModel):
    name: str
    class Config:
        from_attributes = True

class RankResponse(BaseModel):
    name: str
    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    name: str
    department: DepartmentResponse
    rank: RankResponse
    class Config:
        from_attributes = True

# Uygulama Kurulumu
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Uç Noktaları (Endpoints)
@app.get("/departments", response_model=List[DepartmentResponse], tags=["Departmanlar"])
def get_all_departments(db: Session = Depends(get_db)):
    
    return db.query(Department).all()

@app.get("/ranks", response_model=List[RankResponse], tags=["Rütbeler"])
def get_all_ranks(db: Session = Depends(get_db)):
   
    return db.query(Rank).all()

@app.get("/users/query", response_model=List[UserResponse], tags=["Kullanıcılar"])
def query_users(
    department_name: Optional[str] = Query(None, description="Departman adına göre filtrele"),
    rank_name: Optional[str] = Query(None, description="Rütbe adına göre filtrele"),
    db: Session = Depends(get_db)
):
    
    query = db.query(User)

    if department_name:
        query = query.join(Department).filter(Department.name == department_name)
        
    if rank_name:
        query = query.join(Rank).filter(Rank.name == rank_name)

    return query.all()