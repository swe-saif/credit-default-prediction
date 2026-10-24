import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { App } from './app';
import { FEATURES, SAMPLES } from './feature-schema';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  async function render() {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const page = fixture.nativeElement as HTMLElement;
    const submit = async () => {
      page.querySelector('form')!.dispatchEvent(new Event('submit'));
      await fixture.whenStable();
    };
    return { fixture, page, submit };
  }

  it('renders one input per model feature', async () => {
    const { page } = await render();
    expect(page.querySelectorAll('input, select').length).toBe(FEATURES.length);
  });

  it('blocks submission when a required input is blank', async () => {
    const { page, submit } = await render();
    await submit();
    expect(page.querySelector('.error')?.textContent).toContain('Enter a value');
    TestBed.inject(HttpTestingController).expectNone('/api/predict');
  });

  it('sends a sample applicant to the API and shows the result', async () => {
    const { fixture, page, submit } = await render();
    const sampleButton = page.querySelector<HTMLButtonElement>('button.secondary')!;
    sampleButton.click();
    await fixture.whenStable();
    await submit();

    const request = TestBed.inject(HttpTestingController).expectOne('/api/predict');
    expect(request.request.body).toEqual(SAMPLES[0].values);
    request.flush({
      prediction: 1, label: 'Higher default risk', probability: 0.2, probabilities: [0.8, 0.2],
      threshold: 0.138, task: 'classification', model_version: '1.0', disclaimer: 'Prototype only.',
    });
    await fixture.whenStable();
    expect(page.querySelector('.result')?.textContent).toContain('Higher default risk');
    expect(page.querySelector('.result')?.textContent).toContain('20.0%');
  });
});
