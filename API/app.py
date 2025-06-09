# app.py
from fastapi import FastAPI, Query
from kdtree_wrapper import lib, Tarv, TReg
from ctypes import POINTER,c_char, c_float
from pydantic import BaseModel

app = FastAPI()

class FaceEntrada(BaseModel): #Mudanca atualizar elementos de treg
    emb: list[float]
    id: str

@app.get("/")
def home():
    return {"message": "Página inicial do reconhecimento facial utilizando uma base de embeddings armazenados em uma KDTree (ABB modificada)"}

@app.post("/construir-arvore")
def constroi_arvore():
    lib.kdtree_construir()
    return {"mensagem": "Árvore KD inicializada com sucesso."}

@app.post("/inserir")
def inserir(face: FaceEntrada): #Mudanca atuaalizar elementos de treg, funcoes renomeados
    emb_vetor = (c_float * 128)(*face.emb)
    id_bytes = face.id.encode('utf-8')[:99]  # Trunca se necessário
    nova_face = TReg(emb=emb_vetor, id=id_bytes)
    lib.inserir_face(nova_face)
    return {"mensagem": f"Face: '{face.id}' inserida com sucesso."}

@app.post("/buscar") #Atualizar elementos de treg
def buscar(face: FaceEntrada):
    emb_vetor = (c_float * 128)(*face.emb)
    id_bytes = face.id.encode('utf-8')[:99]  # Trunca se necessário
    face_atual = TReg(emb=emb_vetor, id=id_bytes)

    arv = lib.get_tree()  # Suponha que esta função retorne ponteiro para árvore já construída
    resultado = lib.buscar_mais_proximo(arv, face_atual)

    return { #Mudanca atualizar retorno de dados
        "emb": list(resultado.emb), 
        "id": resultado.id
    }