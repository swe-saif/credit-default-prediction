import { Component, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FEATURES, MODEL_INFO, SAMPLES, Sample } from './feature-schema';
import { Payload, PredictionResponse, PredictionService } from './prediction.service';

type FormValue = string | number | null;

const formatNumber = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 }).format;

@Component({
  selector: 'app-root',
  imports: [FormsModule, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private readonly api = inject(PredictionService);

  protected readonly fields = FEATURES;
  protected readonly samples = SAMPLES;
  protected readonly model = MODEL_INFO;
  /** GitHub Pages only serves static files: predictions need Flask, which runs on the user's own machine. */
  protected readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  protected values: Record<string, FormValue> = Object.fromEntries(FEATURES.map((f) => [f.name, null]));
  protected readonly loading = signal(false);
  protected readonly error = signal('');
  protected readonly result = signal<PredictionResponse | null>(null);

  protected fillSample(sample: Sample) {
    this.values = { ...sample.values };
    this.error.set('');
    this.result.set(null);
  }

  protected submit() {
    this.error.set('');
    this.result.set(null);
    if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: start Flask and Angular locally to get predictions.');
      return;
    }
    const payload: Payload = {};
    for (const field of this.fields) {
      const value = this.values[field.name];
      if (value === null || value === '') {
        if (field.required) {
          this.error.set(`Enter a value for ${field.label}`);
          return;
        }
        payload[field.name] = null; // optional: the model treats it as unknown, as in training
        continue;
      }
      if (field.kind === 'number') {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          this.error.set(`Invalid number for ${field.label}`);
          return;
        }
        if (numeric < (field.min ?? -Infinity) || numeric > (field.max ?? Infinity)) {
          this.error.set(`${field.label} must be between ${formatNumber(field.min!)} and ${formatNumber(field.max!)}`);
          return;
        }
        payload[field.name] = numeric;
      } else {
        payload[field.name] = String(value);
      }
    }
    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: (result) => {
        this.result.set(result);
        this.loading.set(false);
      },
      error: (err) => {
        const detail = err.error?.detail ? `: ${[].concat(err.error.detail).join('; ')}` : '';
        this.error.set(`API error ${err.status || ''}: ${err.error?.error ?? 'Is Flask running?'}${detail}`);
        this.loading.set(false);
      },
    });
  }
}
