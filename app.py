import streamlit as st
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Generator
import uuid
import os
import json
from dotenv import load_dotenv
from openai import OpenAI, APIError
import tiktoken

# --- Configuration ---
MODEL = "deepseek/deepseek-r1:free"
BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.7
MAX_TOKENS = 8192
SESSION_FILE = "sessions.json"


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get(
        "OPENROUTER_API_KEY")
    if not api_key:
        st.error("🚨 OpenRouter API key not found!")
        st.stop()
    return api_key

# --- Session Management ---


@dataclass
class Message:
    role: str
    content: str
    tokens: int


@dataclass
class Session:
    key: str
    display_name: str
    messages: List[Message]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SessionManager:
    def __init__(self, tokenizer_encoding: str = "cl100k_base"):
        self._sessions: Dict[str, Session] = {}
        self._active_key: Optional[str] = None
        self._tokenizer = tiktoken.get_encoding(tokenizer_encoding)
        self._load_from_json()
        self._initialize_default_session()

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text)) if self._tokenizer else len(text.split())

    def _generate_display_name(self) -> str:
        base = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        existing = {s.display_name for s in self._sessions.values()}
        counter, name = 1, base
        while name in existing:
            counter += 1
            name = f"{base} ({counter})"
        return name

    def _initialize_default_session(self):
        if not self._sessions:
            key = str(uuid.uuid4())
            self._sessions[key] = Session(
                key=key, display_name=self._generate_display_name(), messages=[])
            self._active_key = key
            self._save_to_json()

    def _load_from_json(self):
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
                for key, session_data in data.items():
                    messages = [Message(**msg)
                                for msg in session_data.pop("messages")]
                    self._sessions[key] = Session(
                        messages=messages, **session_data)
                self._active_key = next(iter(self._sessions), None)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_to_json(self):
        with open(SESSION_FILE, "w") as f:
            json.dump({k: asdict(v) for k, v in self._sessions.items()}, f)

    @property
    def active_key(self) -> Optional[str]:
        if self._active_key not in self._sessions and self._sessions:
            self._active_key = next(iter(self._sessions))
        return self._active_key

    @property
    def active_session(self) -> Optional[Session]:
        return self._sessions.get(self.active_key)

    def get_sessions(self) -> List[Tuple[str, str]]:
        return sorted([(s.key, s.display_name) for s in self._sessions.values()], key=lambda x: x[1])

    def add_session(self) -> str:
        key = str(uuid.uuid4())
        self._sessions[key] = Session(
            key=key, display_name=self._generate_display_name(), messages=[])
        self._active_key = key
        self._save_to_json()
        return key

    def delete_session(self, key: str) -> bool:
        if len(self._sessions) <= 1:
            return False
        if key in self._sessions:
            del self._sessions[key]
            if self._active_key == key:
                self._active_key = next(
                    iter(self._sessions), None) or self.add_session()
            self._save_to_json()
            return True
        return False

    def set_active_session(self, key: str):
        if key in self._sessions:
            self._active_key = key

    def add_message(self, role: str, content: str):
        if session := self.active_session:
            tokens = self._count_tokens(content)
            session.messages.append(
                Message(role=role, content=content, tokens=tokens))
            token_field = "prompt_tokens" if role == "user" else "completion_tokens"
            setattr(session, token_field, getattr(
                session, token_field) + tokens)
            self._save_to_json()

    def get_messages(self) -> List[Message]:
        return self.active_session.messages if self.active_session else []

    def remove_last(self, role: Optional[str] = None):
        if not (session := self.active_session) or not session.messages:
            return
        last = session.messages[-1]
        if role is None or last.role == role:
            session.messages.pop()
            token_field = "prompt_tokens" if last.role == "user" else "completion_tokens"
            setattr(session, token_field, max(
                0, getattr(session, token_field) - last.tokens))
            self._save_to_json()

# --- API Client ---


class APIClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def stream_response(self, messages: List[dict]) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=True
            )
            for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield content
        except APIError as e:
            yield f"Error: {e}"
        except Exception as e:
            yield f"Error: {e}"

# --- UI Components ---


def render_ui(manager: SessionManager, client: APIClient):
    st.set_page_config(layout="wide")
    session = manager.active_session
    if not session:
        st.error("No chat session available.")
        st.stop()

    st.title(f"💬 {session.display_name}")
    st.caption(f"Using Model: {MODEL}")

    for msg in manager.get_messages():
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    with st.sidebar:
        st.header("Chats")
        if st.button("➕ New Chat", use_container_width=True):
            manager.add_session()
            st.rerun()
        
        st.divider()
        expander = st.expander("## Chat Sessions", expanded=False)
        with expander:
            st.subheader("Select Chat")
            for key, name in manager.get_sessions():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.button(name, key=f"select_{key}", on_click=lambda k=key: manager.set_active_session(k),
                              type="primary" if key == manager.active_key else "secondary", use_container_width=True)
                with col2:
                    disabled = len(manager.get_sessions()) <= 1
                    if st.button("❌", key=f"delete_{key}", disabled=disabled, use_container_width=True):
                        if manager.delete_session(key):
                            st.rerun()
                        else:
                            st.toast(
                                "Cannot delete the last session.", icon="⚠️")
        st.divider()
        expander = st.expander("Token Usage", expanded=False)
        with expander:
            st.subheader("Token Usage")
            if session:
                total = session.prompt_tokens + session.completion_tokens
                st.metric("Prompt Tokens", f"{session.prompt_tokens:,}")
                st.metric("Completion Tokens", f"{session.completion_tokens:,}")
                st.metric("Total Tokens", f"{total:,}")

    if prompt := st.chat_input("Enter your prompt here..."):
        manager.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""
            has_error = False
            with st.spinner("🧠 Thinking..."):
                messages = [{"role": m.role, "content": m.content}
                            for m in manager.get_messages()]
                for chunk in client.stream_response(messages):
                    if chunk.startswith("Error:"):
                        has_error = True
                        placeholder.error(chunk[6:].strip())
                        break
                    response += chunk
                    placeholder.markdown(response + "▌")
                if not has_error:
                    placeholder.markdown(response)
                    manager.add_message("assistant", response)
                    st.rerun()
                else:
                    manager.remove_last("user")
                    st.warning("Message not processed due to an error.")

# --- Main ---


def main():
    api_key = load_api_key()
    client = APIClient(api_key)
    manager = st.session_state.setdefault(
        "chat_session_manager", SessionManager())
    render_ui(manager, client)


if __name__ == "__main__":
    main()
