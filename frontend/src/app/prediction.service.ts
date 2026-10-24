import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export type Payload = Record<string, string | number | null>;

/** What the Flask API's POST /predict returns. */
export interface PredictionResponse {
  prediction: 0 | 1;
  label: string;
  probability: number;
  probabilities: number[];
  threshold: number;
  task: string;
  model_version: string;
  disclaimer: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);

  /** Relative URL: the Angular dev server forwards /api/* to Flask (see proxy.conf.json). */
  predict(payload: Payload): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predict', payload);
  }
}
