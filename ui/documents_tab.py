# ui/documents_tab_qt6.py
# -*- coding: utf-8 -*-
"""
PyQt6 'Documents' 탭
- 파일 추가(선택/드래그&드롭)
- 업로드: UPLOAD_DIR로 복사 후 RagStore.upsert_document_files() 호출 → Qdrant 저장
- 인덱스 재빌드: RagStore.upsert_document_dir(UPLOAD_DIR) → 전체 다시 업서트
- 목록 새로고침 / 선택 파일 내보내기 / 업로드 폴더 열기(OS 탐색기)
"""

from __future__ import annotations
import os
import sys
import shutil
import threading
import traceback
import subprocess
from typing import List

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel,
    QFileDialog, QProgressBar, QMessageBox
)

# RAG Store (Qdrant 통합 파이프라인)
from core.rag.rag_store import RagStore

# 업로드 폴더(프로젝트 표준 경로로 고정)
UPLOAD_DIR = os.path.join("data", "docs", "uploaded")


class DocumentsTab(QWidget):
    # 백그라운드 작업용 시그널
    log_sig  = Signal(str)
    prog_sig = Signal(int)
    done_sig = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.setAcceptDrops(True)

        # 업로드 대기열(풀경로)
        self.queue: List[str] = []
        # RAG Store (Qdrant 로컬 DB 사용)
        self.store = RagStore(persist_path="./qdrant_db")

        # ---------- UI ----------
        v = QVBoxLayout(self)

        self.lbl = QLabel(
            "문서(.txt/.md/.docx/.pdf/.hwpx)를 추가하세요.\n"
        )
        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl)

        self.list = QListWidget()
        self.list.setSelectionMode(self.list.SelectionMode.ExtendedSelection)
        v.addWidget(self.list)

        row1 = QHBoxLayout()
        self.btn_add = QPushButton("파일 추가")
        self.btn_del = QPushButton("선택 삭제(대기열)")
        self.btn_refresh = QPushButton("목록 새로고침")
        self.btn_open = QPushButton("업로드 폴더 열기")
        row1.addWidget(self.btn_add)
        row1.addWidget(self.btn_del)
        row1.addWidget(self.btn_refresh)
        row1.addStretch(1)
        row1.addWidget(self.btn_open)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_upload = QPushButton("업로드 실행(복사→Qdrant)")
        self.btn_reindex = QPushButton("인덱스 재빌드(전체 재업서트)")
        self.btn_export = QPushButton("선택 파일 내보내기")
        row2.addWidget(self.btn_upload)
        row2.addWidget(self.btn_reindex)
        row2.addStretch(1)
        row2.addWidget(self.btn_export)
        v.addLayout(row2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        self.status = QLabel("대기")
        v.addWidget(self.status)

        # 시그널 연결
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_del.clicked.connect(self.on_del_selected)
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_upload.clicked.connect(self.on_upload)
        self.btn_reindex.clicked.connect(self.on_reindex)
        self.btn_export.clicked.connect(self.on_export)

        self.log_sig.connect(self.status.setText)
        self.prog_sig.connect(self.progress.setValue)
        self.done_sig.connect(self._on_done)

        # 최초 목록
        self.refresh_list()

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.isfile(p) and p not in self.queue:
                self.queue.append(p)
                self.list.addItem(os.path.basename(p) + "  (대기)")

    # ---------- 목록/파일 선택 ----------
    def refresh_list(self):
        self.list.clear()
        for name in sorted(os.listdir(UPLOAD_DIR)):
            path = os.path.join(UPLOAD_DIR, name)
            if os.path.isfile(path):
                self.list.addItem(name)
        self.status.setText("목록 새로고침 완료")

    def on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "문서 선택", "", "Docs (*.txt *.md *.docx *.pdf *.hwpx);;All (*.*)"
        )
        for p in paths:
            if p and p not in self.queue:
                self.queue.append(p)
                self.list.addItem(os.path.basename(p) + "  (대기)")

    def on_del_selected(self):
        for it in self.list.selectedItems():
            base = it.text().replace("  (대기)", "")
            # 대기열에서만 제거(실제 업로드 파일은 보수적으로 유지)
            self.queue = [q for q in self.queue if os.path.basename(q) != base]
            self.list.takeItem(self.list.row(it))

    # ---------- 업로드(복사 → Qdrant 업서트) ----------
    def on_upload(self):
        if not self.queue:
            QMessageBox.information(self, "알림", "대기열이 비었습니다.")
            return
        threading.Thread(target=self._worker_upload, daemon=True).start()

    def _worker_upload(self):
        try:
            self.prog_sig.emit(0)
            self.log_sig.emit("업로드 시작…")

            copied_paths: List[str] = []
            for i, src in enumerate(self.queue, start=1):
                dst = os.path.join(UPLOAD_DIR, os.path.basename(src))
                shutil.copy2(src, dst)
                copied_paths.append(dst)
                self.log_sig.emit(f"복사 완료: {os.path.basename(src)}")
                self.prog_sig.emit(int(i / max(1, len(self.queue)) * 50))  # 0~50%

            # Qdrant에 문서 청크 업서트
            self.log_sig.emit("Qdrant 업서트 중…")
            added = self.store.upsert_document_files(copied_paths, project="Persona-AI", tags=["uploaded"])
            self.prog_sig.emit(90)

            self.queue.clear()
            self.refresh_list()
            self.prog_sig.emit(100)
            self.done_sig.emit(f"✅ 업로드 완료: {len(copied_paths)}개 파일, {added} 청크 업서트")
        except Exception as e:
            self.done_sig.emit(f"❌ 업로드 실패: {e}\n{traceback.format_exc()}")

    # ---------- 인덱스 재빌드(전체 업로드 폴더 재업서트) ----------
    def on_reindex(self):
        threading.Thread(target=self._worker_reindex, daemon=True).start()

    def _worker_reindex(self):
        try:
            self.prog_sig.emit(0)
            self.log_sig.emit("업로드 폴더 스캔…")
            added = self.store.upsert_document_dir(UPLOAD_DIR)
            self.prog_sig.emit(100)
            self.done_sig.emit(f"✅ 인덱스 재빌드 완료: {added} 청크 업서트")
        except Exception as e:
            self.done_sig.emit(f"❌ 재빌드 실패: {e}\n{traceback.format_exc()}")

    # ---------- 선택 파일 내보내기 ----------
    def on_export(self):
        items = self.list.selectedItems()
        if not items:
            QMessageBox.information(self, "알림", "내보낼 파일을 선택하세요.")
            return

        outdir = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if not outdir:
            return

        cnt = 0
        try:
            for it in items:
                name = it.text().replace("  (대기)", "")
                src = os.path.join(UPLOAD_DIR, name)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(outdir, name))
                    cnt += 1
            QMessageBox.information(self, "완료", f"{cnt}개 파일 저장됨.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"다운로드 실패: {e}")

    # ---------- 업로드 폴더를 OS 기본 탐색기로 열기 ----------
    def open_folder(self):
        path = os.path.abspath(UPLOAD_DIR)
        try:
            if os.name == "nt":           # Windows
                os.startfile(path)
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", path])
            else:                         # Linux (POSIX)
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "오류", f"폴더 열기 실패: {e}")

    @Slot(str)
    def _on_done(self, msg: str):
        self.status.setText(msg)
        QMessageBox.information(self, "결과", msg)
