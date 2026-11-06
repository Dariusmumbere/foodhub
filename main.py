"""
Local Restaurant Food Delivery API: single-file FastAPI + Neon PostgreSQL.

Roles: customer, owner (restaurant), rider (company-employed), admin (dispatcher)
Run:   uvicorn main:app --reload --host 0.0.0.0
Env:   DATABASE_URL (Neon), SECRET_KEY
"""
import math
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import bcrypt
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, select)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)

# ───────────────────────── Config ─────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
TOKEN_HOURS = 24 * 7
# Distance-based delivery fee (UGX)
FEE_PER_KM = int(os.getenv("FEE_PER_KM", "2000"))
MIN_FEE = int(os.getenv("MIN_FEE", "2000"))            # anything under 1 km still pays this
FEE_ROUND_TO = int(os.getenv("FEE_ROUND_TO", "500"))   # round up to nearest 500 for easy cash change
MAX_DELIVERY_KM = float(os.getenv("MAX_DELIVERY_KM", "15"))
DETOUR_FACTOR = float(os.getenv("DETOUR_FACTOR", "1.0"))  # straight-line -> road estimate (e.g. 1.3)
ADMIN_PHONE = os.getenv("ADMIN_PHONE", "")        # seeds the first admin on startup
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# pool_pre_ping + recycle: Neon suspends idle compute, so stale connections happen
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def now():
    return datetime.now(timezone.utc)


# ───────────────────────── Models ─────────────────────────
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(120), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), default="customer")  # customer|owner|rider|admin
    is_on_duty: Mapped[bool] = mapped_column(Boolean, default=False)  # riders only
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Restaurant(Base):
    __tablename__ = "restaurants"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    address: Mapped[str] = mapped_column(String(200), default="")
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    phone: Mapped[str] = mapped_column(String(20), default="")
    image_url: Mapped[str] = mapped_column(String(300), default="")
    commission_pct: Mapped[int] = mapped_column(Integer, default=10)  # platform cut of subtotal
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True)  # admin vets restaurants
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    menu: Mapped[List["MenuItem"]] = relationship(back_populates="restaurant", lazy="selectin")


class MenuItem(Base):
    __tablename__ = "menu_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    restaurant_id: Mapped[int] = mapped_column(ForeignKey("restaurants.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    price: Mapped[int] = mapped_column(Integer)  # UGX, whole shillings
    image_url: Mapped[str] = mapped_column(String(300), default="")
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    restaurant: Mapped[Restaurant] = relationship(back_populates="menu")


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    restaurant_id: Mapped[int] = mapped_column(ForeignKey("restaurants.id"), index=True)
    rider_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    delivery_address: Mapped[str] = mapped_column(String(300))
    delivery_lat: Mapped[float] = mapped_column(Float)
    delivery_lng: Mapped[float] = mapped_column(Float)
    distance_km: Mapped[float] = mapped_column(Float)
    note: Mapped[str] = mapped_column(Text, default="")
    payment_method: Mapped[str] = mapped_column(String(20), default="cash")  # cash|mobile_money
    subtotal: Mapped[int] = mapped_column(Integer)
    delivery_fee: Mapped[int] = mapped_column(Integer)
    total: Mapped[int] = mapped_column(Integer)
    commission: Mapped[int] = mapped_column(Integer, default=0)
    cash_settled: Mapped[bool] = mapped_column(Boolean, default=False)  # rider handed cash to office
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    items: Mapped[List["OrderItem"]] = relationship(lazy="selectin")


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    menu_item_id: Mapped[int] = mapped_column(ForeignKey("menu_items.id"))
    name: Mapped[str] = mapped_column(String(120))  # snapshot, prices can change later
    price: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)


# ───────────────────────── Schemas ─────────────────────────
class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RegisterIn(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    password: str = Field(min_length=6)
    role: str = "customer"  # customer|owner (riders/admins are created by admin)


class LoginIn(BaseModel):
    phone: str
    password: str


class UserOut(ORM):
    id: int
    name: str
    phone: str
    email: Optional[str]
    role: str


class TokenOut(BaseModel):
    access_token: str
    user: UserOut


class RestaurantIn(BaseModel):
    name: str
    description: str = ""
    address: str = ""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    phone: str = ""
    image_url: str = ""
    is_open: bool = True


class MenuItemIn(BaseModel):
    name: str
    description: str = ""
    price: int = Field(gt=0)
    image_url: str = ""
    is_available: bool = True


class MenuItemOut(ORM, MenuItemIn):
    id: int
    restaurant_id: int


class RestaurantOut(ORM, RestaurantIn):
    id: int
    owner_id: int
    is_approved: bool
    commission_pct: int


class RestaurantDetail(RestaurantOut):
    menu: List[MenuItemOut] = []


class CartLine(BaseModel):
    menu_item_id: int
    quantity: int = Field(gt=0, le=50)


class OrderIn(BaseModel):
    restaurant_id: int
    items: List[CartLine] = Field(min_length=1)
    delivery_address: str
    delivery_lat: float = Field(ge=-90, le=90)
    delivery_lng: float = Field(ge=-180, le=180)
    note: str = ""
    payment_method: str = "cash"


class OrderItemOut(ORM):
    id: int
    menu_item_id: int
    name: str
    price: int
    quantity: int


class OrderOut(ORM):
    id: int
    customer_id: int
    restaurant_id: int
    rider_id: Optional[int]
    status: str
    delivery_address: str
    delivery_lat: float
    delivery_lng: float
    distance_km: float
    note: str
    payment_method: str
    subtotal: int
    delivery_fee: int
    total: int
    created_at: datetime
    items: List[OrderItemOut]


class StatusIn(BaseModel):
    status: str


# ───────────────────────── Auth helpers ─────────────────────────
bearer = HTTPBearer()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def make_token(user: User) -> str:
    payload = {"sub": str(user.id), "exp": now() + timedelta(hours=TOKEN_HOURS)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)
) -> User:
    try:
        data = jwt.decode(creds.credentials, SECRET_KEY, algorithms=["HS256"])
        user = db.get(User, int(data["sub"]))
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    if not user:
        raise HTTPException(401, "User not found")
    return user


def require_role(*roles: str):
    def checker(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, f"Requires role: {', '.join(roles)}")
        return user
    return checker


def owned_restaurant(db: Session, restaurant_id: int, user: User) -> Restaurant:
    r = db.get(Restaurant, restaurant_id)
    if not r:
        raise HTTPException(404, "Restaurant not found")
    if r.owner_id != user.id:
        raise HTTPException(403, "Not your restaurant")
    return r


# ───────────────────────── Delivery fee ─────────────────────────
def road_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in km, scaled by DETOUR_FACTOR to approximate road distance."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * 6371.0 * math.asin(math.sqrt(a)) * DETOUR_FACTOR, 2)


def delivery_fee_for(km: float) -> int:
    if km > MAX_DELIVERY_KM:
        raise HTTPException(400, f"Too far: we deliver up to {MAX_DELIVERY_KM:g} km")
    fee = math.ceil(km * FEE_PER_KM / FEE_ROUND_TO) * FEE_ROUND_TO
    return max(fee, MIN_FEE)


# ───────────────────────── App ─────────────────────────
app = FastAPI(title="Local Food Delivery API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def init_db():
    Base.metadata.create_all(engine)  # swap for Alembic once schema stabilises
    if ADMIN_PHONE and ADMIN_PASSWORD:
        with SessionLocal() as db:
            if not db.scalar(select(User).where(User.phone == ADMIN_PHONE)):
                db.add(User(name="Admin", phone=ADMIN_PHONE, role="admin",
                            password_hash=bcrypt.hashpw(ADMIN_PASSWORD.encode(),
                                                        bcrypt.gensalt()).decode()))
                db.commit()


@app.get("/")
def health():
    return {"status": "ok"}


# ── Auth ──
@app.post("/auth/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if body.role not in ("customer", "owner"):
        raise HTTPException(400, "Invalid role")
    if db.scalar(select(User).where(User.phone == body.phone)):
        raise HTTPException(409, "Phone already registered")
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    user = User(name=body.name, phone=body.phone, email=body.email,
                password_hash=hashed, role=body.role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": make_token(user), "user": user}


@app.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.phone == body.phone))
    if not user or not bcrypt.checkpw(body.password.encode(), user.password_hash.encode()):
        raise HTTPException(401, "Wrong phone or password")
    return {"access_token": make_token(user), "user": user}


@app.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


# ── Restaurants (public browse) ──
@app.get("/restaurants", response_model=List[RestaurantOut])
def list_restaurants(q: Optional[str] = None, db: Session = Depends(get_db)):
    stmt = select(Restaurant).where(Restaurant.is_open.is_(True), Restaurant.is_approved.is_(True))
    if q:
        stmt = stmt.where(Restaurant.name.ilike(f"%{q}%"))
    return db.scalars(stmt).all()


@app.get("/restaurants/{rid}", response_model=RestaurantDetail)
def get_restaurant(rid: int, db: Session = Depends(get_db)):
    r = db.get(Restaurant, rid)
    if not r:
        raise HTTPException(404, "Restaurant not found")
    return r


# ── Restaurants (owner) ──
@app.post("/restaurants", response_model=RestaurantOut, status_code=201)
def create_restaurant(body: RestaurantIn, user: User = Depends(require_role("owner")),
                      db: Session = Depends(get_db)):
    r = Restaurant(owner_id=user.id, **body.model_dump())
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


@app.get("/owner/restaurants", response_model=List[RestaurantOut])
def my_restaurants(user: User = Depends(require_role("owner")), db: Session = Depends(get_db)):
    return db.scalars(select(Restaurant).where(Restaurant.owner_id == user.id)).all()


@app.put("/restaurants/{rid}", response_model=RestaurantOut)
def update_restaurant(rid: int, body: RestaurantIn, user: User = Depends(require_role("owner")),
                      db: Session = Depends(get_db)):
    r = owned_restaurant(db, rid, user)
    for k, v in body.model_dump().items():
        setattr(r, k, v)
    db.commit()
    db.refresh(r)
    return r


# ── Menu (owner) ──
@app.post("/restaurants/{rid}/menu", response_model=MenuItemOut, status_code=201)
def add_menu_item(rid: int, body: MenuItemIn, user: User = Depends(require_role("owner")),
                  db: Session = Depends(get_db)):
    owned_restaurant(db, rid, user)
    item = MenuItem(restaurant_id=rid, **body.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.put("/menu/{item_id}", response_model=MenuItemOut)
def update_menu_item(item_id: int, body: MenuItemIn, user: User = Depends(require_role("owner")),
                     db: Session = Depends(get_db)):
    item = db.get(MenuItem, item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    owned_restaurant(db, item.restaurant_id, user)
    for k, v in body.model_dump().items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item


@app.delete("/menu/{item_id}", status_code=204)
def delete_menu_item(item_id: int, user: User = Depends(require_role("owner")),
                     db: Session = Depends(get_db)):
    item = db.get(MenuItem, item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    owned_restaurant(db, item.restaurant_id, user)
    item.is_available = False  # soft delete keeps old orders intact
    db.commit()


# ── Orders ──
@app.post("/orders", response_model=OrderOut, status_code=201)
def place_order(body: OrderIn, user: User = Depends(require_role("customer")),
                db: Session = Depends(get_db)):
    r = db.get(Restaurant, body.restaurant_id)
    if not r or not r.is_open or not r.is_approved:
        raise HTTPException(400, "Restaurant unavailable")
    if body.payment_method not in ("cash", "mobile_money"):
        raise HTTPException(400, "Invalid payment method")

    lines, subtotal = [], 0
    for line in body.items:
        mi = db.get(MenuItem, line.menu_item_id)
        if not mi or mi.restaurant_id != r.id or not mi.is_available:
            raise HTTPException(400, f"Item {line.menu_item_id} unavailable")
        subtotal += mi.price * line.quantity
        lines.append(OrderItem(menu_item_id=mi.id, name=mi.name,
                               price=mi.price, quantity=line.quantity))

    km = road_km(r.latitude, r.longitude, body.delivery_lat, body.delivery_lng)
    fee = delivery_fee_for(km)  # computed server-side so the app can't tamper with it
    order = Order(customer_id=user.id, restaurant_id=r.id,
                  delivery_lat=body.delivery_lat, delivery_lng=body.delivery_lng,
                  distance_km=km,
                  delivery_address=body.delivery_address, note=body.note,
                  payment_method=body.payment_method, subtotal=subtotal,
                  delivery_fee=fee, total=subtotal + fee,
                  commission=subtotal * r.commission_pct // 100,
                  items=lines)  # prices computed server-side, never trust the client
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


@app.get("/orders/mine", response_model=List[OrderOut])
def my_orders(user: User = Depends(require_role("customer")), db: Session = Depends(get_db)):
    return db.scalars(select(Order).where(Order.customer_id == user.id)
                      .order_by(Order.created_at.desc())).all()


@app.get("/restaurants/{rid}/orders", response_model=List[OrderOut])
def restaurant_orders(rid: int, user: User = Depends(require_role("owner")),
                      db: Session = Depends(get_db)):
    owned_restaurant(db, rid, user)
    return db.scalars(select(Order).where(Order.restaurant_id == rid)
                      .order_by(Order.created_at.desc())).all()


@app.get("/orders/{oid}", response_model=OrderOut)
def get_order(oid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    o = db.get(Order, oid)
    if not o:
        raise HTTPException(404, "Order not found")
    r = db.get(Restaurant, o.restaurant_id)
    if user.id not in (o.customer_id, o.rider_id, r.owner_id):
        raise HTTPException(403, "Not your order")
    return o


OWNER_STATUSES = {"accepted", "preparing", "ready", "cancelled"}
RIDER_STATUSES = {"out_for_delivery", "delivered"}


@app.patch("/orders/{oid}/status", response_model=OrderOut)
def update_status(oid: int, body: StatusIn, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    o = db.get(Order, oid)
    if not o:
        raise HTTPException(404, "Order not found")
    r = db.get(Restaurant, o.restaurant_id)
    s = body.status

    if user.role == "admin":
        pass  # dispatcher can fix any order
    elif user.role == "owner" and r.owner_id == user.id and s in OWNER_STATUSES:
        pass
    elif user.role == "rider" and o.rider_id == user.id and s in RIDER_STATUSES:
        pass
    elif user.role == "customer" and o.customer_id == user.id and s == "cancelled" \
            and o.status == "pending":
        pass
    else:
        raise HTTPException(403, "Not allowed to set this status")

    if o.status in ("delivered", "cancelled"):
        raise HTTPException(400, f"Order already {o.status}")
    o.status = s
    db.commit()
    db.refresh(o)
    return o


# ── Rider ──
@app.get("/rider/available", response_model=List[OrderOut])
def available_orders(user: User = Depends(require_role("rider")), db: Session = Depends(get_db)):
    return db.scalars(select(Order).where(Order.status == "ready", Order.rider_id.is_(None))
                      .order_by(Order.created_at)).all()


@app.post("/orders/{oid}/claim", response_model=OrderOut)
def claim_order(oid: int, user: User = Depends(require_role("rider")),
                db: Session = Depends(get_db)):
    o = db.scalar(select(Order).where(Order.id == oid).with_for_update())  # avoid double-claim
    if not user.is_on_duty:
        raise HTTPException(400, "Go on duty first")
    if not o or o.status != "ready" or o.rider_id is not None:
        raise HTTPException(400, "Order not available")
    o.rider_id = user.id
    db.commit()
    db.refresh(o)
    return o


@app.get("/rider/mine", response_model=List[OrderOut])
def rider_orders(user: User = Depends(require_role("rider")), db: Session = Depends(get_db)):
    return db.scalars(select(Order).where(Order.rider_id == user.id)
                      .order_by(Order.created_at.desc())).all()


class DutyIn(BaseModel):
    on_duty: bool


@app.post("/rider/duty", response_model=UserOut)
def set_duty(body: DutyIn, user: User = Depends(require_role("rider")),
             db: Session = Depends(get_db)):
    user.is_on_duty = body.on_duty
    db.commit()
    db.refresh(user)
    return user


# ───────────────────────── Admin / dispatch ─────────────────────────
class RiderIn(BaseModel):
    name: str
    phone: str
    password: str = Field(min_length=6)


class RestaurantAdminIn(BaseModel):
    is_approved: bool
    commission_pct: int = Field(ge=0, le=50, default=10)


@app.post("/admin/riders", response_model=UserOut, status_code=201)
def create_rider(body: RiderIn, _: User = Depends(require_role("admin")),
                 db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.phone == body.phone)):
        raise HTTPException(409, "Phone already registered")
    rider = User(name=body.name, phone=body.phone, role="rider",
                 password_hash=bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode())
    db.add(rider)
    db.commit()
    db.refresh(rider)
    return rider


@app.get("/admin/riders", response_model=List[UserOut])
def list_riders(_: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    return db.scalars(select(User).where(User.role == "rider")).all()


@app.get("/admin/restaurants", response_model=List[RestaurantOut])
def admin_restaurants(_: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    return db.scalars(select(Restaurant).order_by(Restaurant.is_approved, Restaurant.id)).all()


@app.patch("/admin/restaurants/{rid}", response_model=RestaurantOut)
def approve_restaurant(rid: int, body: RestaurantAdminIn,
                       _: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    r = db.get(Restaurant, rid)
    if not r:
        raise HTTPException(404, "Restaurant not found")
    r.is_approved, r.commission_pct = body.is_approved, body.commission_pct
    db.commit()
    db.refresh(r)
    return r


@app.get("/admin/orders", response_model=List[OrderOut])
def admin_orders(status: Optional[str] = None, _: User = Depends(require_role("admin")),
                 db: Session = Depends(get_db)):
    stmt = select(Order).order_by(Order.created_at.desc()).limit(200)
    if status:
        stmt = stmt.where(Order.status == status)
    return db.scalars(stmt).all()


@app.post("/admin/orders/{oid}/assign/{rider_id}", response_model=OrderOut)
def assign_rider(oid: int, rider_id: int, _: User = Depends(require_role("admin")),
                 db: Session = Depends(get_db)):
    o, rider = db.get(Order, oid), db.get(User, rider_id)
    if not o or not rider or rider.role != "rider":
        raise HTTPException(404, "Order or rider not found")
    if o.status in ("delivered", "cancelled"):
        raise HTTPException(400, f"Order already {o.status}")
    o.rider_id = rider.id  # may be assigned before "ready" so the rider arrives on time
    db.commit()
    db.refresh(o)
    return o


def _unsettled(db: Session, rider_id: int):
    return db.scalars(select(Order).where(
        Order.rider_id == rider_id, Order.status == "delivered",
        Order.payment_method == "cash", Order.cash_settled.is_(False))).all()


@app.get("/admin/riders/{rider_id}/cash")
def rider_cash_owed(rider_id: int, _: User = Depends(require_role("admin")),
                    db: Session = Depends(get_db)):
    orders = _unsettled(db, rider_id)
    return {"rider_id": rider_id, "orders": len(orders), "owed": sum(o.total for o in orders)}


@app.post("/admin/riders/{rider_id}/settle")
def settle_rider_cash(rider_id: int, _: User = Depends(require_role("admin")),
                      db: Session = Depends(get_db)):
    orders = _unsettled(db, rider_id)
    total = sum(o.total for o in orders)
    for o in orders:
        o.cash_settled = True
    db.commit()
    return {"rider_id": rider_id, "settled_orders": len(orders), "amount": total}


# ── Delivery quote (call this before checkout to show the fee) ──
@app.get("/delivery-quote")
def delivery_quote(restaurant_id: int, lat: float, lng: float, db: Session = Depends(get_db)):
    r = db.get(Restaurant, restaurant_id)
    if not r:
        raise HTTPException(404, "Restaurant not found")
    km = road_km(r.latitude, r.longitude, lat, lng)
    return {"distance_km": km, "delivery_fee": delivery_fee_for(km)}
