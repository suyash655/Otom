import { useState, useCallback } from "react";
import { predict, explain, ExplainResult, PredictionResult } from "../../lib/api";

export function usePrediction() {
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>("");

  const predictImage = useCallback(async (file: File) => {
    setStatus("loading");
    setError("");
    try {
      const data = await predict(file);
      setResult(data);
      setStatus("done");
    } catch (e: unknown) {
      setError((e as Error).message ?? "Unknown error");
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError("");
  }, []);

  return { status, result, error, predictImage, reset };
}

export function useExplanation() {
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [result, setResult] = useState<ExplainResult | null>(null);
  const [error, setError] = useState<string>("");

  const explainImage = useCallback(async (file: File) => {
    setStatus("loading");
    setError("");
    try {
      const data = await explain(file);
      setResult(data);
      setStatus("done");
    } catch (e: unknown) {
      setError((e as Error).message ?? "Unknown error");
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError("");
  }, []);

  return { status, result, error, explainImage, reset };
}