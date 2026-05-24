import { useState, useRef, ClipboardEvent, KeyboardEvent } from "react";
import { Send, Paperclip } from "lucide-react";
import { Button } from "./ui/button";

interface ChatInputProps {
  onSend: (message: string) => void;
  placeholder?: string;
  isMainChat?: boolean;
}

export const ChatInput = ({ onSend, placeholder = "Type your message...", isMainChat = false }: ChatInputProps) => {
  const editableDivRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const content = editableDivRef.current?.innerHTML.trim();
    if (content) {
      onSend(content);
      if (editableDivRef.current) {
        editableDivRef.current.innerHTML = "";
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (isMainChat && e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handlePaste = (e: ClipboardEvent<HTMLDivElement>) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        e.preventDefault();
        const file = items[i].getAsFile();
        if (!file) return;
        appendImageFromFile(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      appendImageFromFile(file);
    }
    e.target.value = ""; // Reset file input
  };

  const appendImageFromFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = document.createElement("img");
      img.src = event.target?.result as string;
      img.style.maxWidth = "100%";
      img.style.maxHeight = "100px";
      img.style.borderRadius = "4px";
      editableDivRef.current?.appendChild(img);
    };
    reader.readAsDataURL(file);
  };

  const handleInput = (e: React.FormEvent<HTMLDivElement>) => {
    // This function is kept to handle potential future logic on input
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t border-border bg-card/50 backdrop-blur-sm">
       {isMainChat && (
        <>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="self-end"
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip className="h-5 w-5" />
          </Button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            accept="image/*"
          />
        </>
      )}
      <div
        ref={editableDivRef}
        contentEditable="true"
        onInput={handleInput}
        onPaste={handlePaste}
        onKeyDown={handleKeyDown}
        data-placeholder={placeholder}
        className="flex-1 bg-input border-terminal-green/30 focus:border-terminal-green terminal-glow text-foreground p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-terminal-green/50 min-h-[40px] max-h-48 overflow-y-auto empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground"
      />
      <Button
        type="submit"
        size="icon"
        className="bg-terminal-green hover:bg-terminal-glow text-primary-foreground terminal-glow self-end"
      >
        <Send className="h-4 w-4" />
      </Button>
    </form>
  );
};
