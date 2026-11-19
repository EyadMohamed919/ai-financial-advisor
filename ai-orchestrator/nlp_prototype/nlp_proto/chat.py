"""Try the pipeline from the terminal.

Usage:
  python -m nlp_proto.chat                      interactive
  python -m nlp_proto.chat "دفعت 120 أكل" ...   one reply per argument
"""
import json
import sys

from .pipeline import Assistant


def show(assistant, text):
    out = assistant.handle(text)
    print(f"\n> {text}\n{out['reply']}\n{json.dumps(out['extracted_data'], ensure_ascii=False)}  [{out['status']}]")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    assistant = Assistant()
    if len(sys.argv) > 1:
        for text in sys.argv[1:]:
            show(assistant, text)
        return
    print("اكتب رسالة (Ctrl+C للخروج)")
    while True:
        try:
            show(assistant, input("\n> "))
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
