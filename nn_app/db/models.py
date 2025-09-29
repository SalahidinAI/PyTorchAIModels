from .database import Base
from sqlalchemy import String, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column


class Mnist(Base):
    __tablename__ = 'mnist'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[int] = mapped_column(Integer)


class Fashion(Base):
    __tablename__ = 'fashion'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Cifar100(Base):
    __tablename__ = 'cifar100'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class SpeechCommands(Base):
    __tablename__ = 'speech_commands'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Gtzan(Base):
    __tablename__ = 'gtzan'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Urban(Base):
    __tablename__ = 'urban'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class News(Base):
    __tablename__ = 'news'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String)
