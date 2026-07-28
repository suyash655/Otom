import { useState, useRef, useCallback } from "react";

export function useFileHandler() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFile = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) loadFile(f);
  }, [loadFile]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) loadFile(f);
  }, [loadFile]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  const reset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setDragging(false);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }, []);

  const triggerFileSelect = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return {
    file,
    preview,
    dragging,
    inputRef,
    onFileChange,
    onDrop,
    onDragOver,
    onDragLeave,
    reset,
    triggerFileSelect,
  };
}