"""Fatiamento das fontes em unidades de recuperação."""

from .extracao import Bloco, extrair
from .juntas import Junta, capitulo, detectar, marco, subtitulo, sucede
from .pedacos import TETO, Pedaco, fatiar, livro_de_capitulo

__all__ = [
    "Bloco",
    "Junta",
    "Pedaco",
    "TETO",
    "capitulo",
    "detectar",
    "extrair",
    "fatiar",
    "livro_de_capitulo",
    "marco",
    "subtitulo",
    "sucede",
]
