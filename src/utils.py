"""Notebook utils."""

import json
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def save_docs_to_json(array: Iterable[Document], file_path: str) -> None:
    path = Path(file_path)
    with open(path, "w", encoding="utf-8") as json_file:
        for doc in array:
            json_file.write(doc.json() + "\n")


def load_docs_from_json(file_path: str) -> Iterable[Document]:
    path = Path(file_path)
    array = []
    with open(path, "r", encoding="utf-8") as json_file:
        for line in json_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
