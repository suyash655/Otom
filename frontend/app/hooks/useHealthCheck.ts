import { useState, useEffect } from "react";
import { healthCheck } from "../../lib/api";
import { HEALTH_CHECK_INTERVAL } from "../../lib/constants";

export function useHealthCheck(interval: number = HEALTH_CHECK_INTERVAL) {
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    const check = () => {
      healthCheck().then(setApiOk);
    };

    // Initial check
    check();

    // Periodic checks
    const intervalId = setInterval(check, interval);
    return () => clearInterval(intervalId);
  }, [interval]);

  return apiOk;
}