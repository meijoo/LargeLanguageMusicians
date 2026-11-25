"""Run ComposerSingle and NegotiatingComposerAgent as two complete composers."""
import argparse
import json
import os
from pathlib import Path

from ComposerSingle import composer as single
from .agent import ComposerResponse, InvalidComposerResponse, NegotiatingComposerAgent


def run(prompt, output, full_six_turns=False):
    output.mkdir(parents=True, exist_ok=True)
    reuse_initial = False
    if (output / 'negotiation.json').exists():
        prior = json.loads((output / 'negotiation.json').read_text())
        reuse_initial = (prior.get('status') == 'failed' and prior.get('prompt') == prompt
                         and [entry['phase'] for entry in prior.get('exchanges', [])] == ['independent draft']
                         and prior['exchanges'][0]['speaker'] == 'ComposerSingle')
        if prior.get('status') != 'failed' or (prior.get('exchanges') and not reuse_initial):
            raise ValueError('Choose a new output directory to preserve the existing negotiation.')
        archive = output / 'failed_attempts'
        archive.mkdir(exist_ok=True)
        index = len(list(archive.glob('*.json'))) + 1
        (archive / f'attempt_{index}.json').write_text(json.dumps(prior, indent=2) + '\n')
    single.load_dotenv()
    model = os.environ.get('AZURE_OPENAI_DEPLOYMENT') or os.environ.get('OPENAI_MODEL')
    key = os.environ.get('AZURE_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT') or os.environ.get('OPENAI_ENDPOINT')
    base = endpoint.rstrip('/') + '/openai/v1' if endpoint else os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    if not model or not key:
        raise ValueError('Configure a model and API key in .env.')
    log = dict(prompt=prompt, model=model, status='started', full_six_turns=full_six_turns, exchanges=[])
    history = []

    def save():
        (output / 'negotiation.json').write_text(json.dumps(log, indent=2) + '\n')
        paragraphs = [f'# Composer negotiation\n\nPrompt: {prompt}\n\nStatus: {log["status"]}']
        for entry in log['exchanges']:
            response = entry['response']
            paragraphs.append(f'## {entry["phase"]}: {entry["speaker"]}\n\n{response["message"]}\n\n'
                              + '\n'.join('- ' + item for item in response['plan'])
                              + f'\n\nAccepts peer plan: {response["accepts_peer_plan"]}')
        (output / 'negotiation.md').write_text('\n\n'.join(paragraphs) + '\n')
        (output / 'single_history.json').write_text(json.dumps(history, indent=2) + '\n')

    def record(speaker, phase, response):
        log['exchanges'].append(dict(speaker=speaker, phase=phase, response=json.loads(response.to_json())))
        save()
        print(f'{phase}: {speaker}', flush=True)

    def peer_request(method, *args, **kwargs):
        for attempt in range(3):
            try:
                return method(*args, **kwargs)
            except InvalidComposerResponse as error:
                log.setdefault('invalid_peer_responses', []).append(dict(
                    attempt=attempt + 1, turn=kwargs.get('turn'),
                    validation_error=str(error), raw_response=error.raw_response))
                save()
                if attempt == 2:
                    raise

    def single_reply(instruction):
        nonlocal history
        messages = history + [{'role': 'user', 'content': instruction}]
        for attempt in range(3):
            raw = single.request_completion(messages, model, key, base)
            (output / 'single_latest_raw.txt').write_text(raw)
            try:
                data = json.loads(raw)
                score = single.validate_score(data['score'], 16)
                response = ComposerResponse.from_json(json.dumps(dict(
                    message=data['message'], plan=data['plan'], abc=single.to_abc(score),
                    accepts_peer_plan=data['accepts_peer_plan'])))
                history = messages + [{'role': 'assistant', 'content': raw}]
                return response, score
            except (ValueError, KeyError, TypeError) as error:
                if attempt == 2:
                    raise
                messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content':
                    f'Invalid response: {error}. Return the complete corrected JSON wrapper and score.'}]

    save()
    try:
        peer = NegotiatingComposerAgent(name='NegotiatingComposer')
        if reuse_initial:
            score = single.validate_score(json.loads((output / 'initial_single' / 'composition.json').read_text()), 16)
            history = json.loads((output / 'initial_single' / 'run.json').read_text())['messages']
        else:
            score, history = single.compose(prompt, model=model, api_key=key, base_url=base)
        initial = ComposerResponse('Independent ComposerSingle draft; proposed musical details are in the complete score.',
                                   (f'Use the complete 16-bar piano draft at {score["tempo"]} BPM in {score["key"]}; refine details with the peer.',),
                                   single.to_abc(score), False)
        single.export(score, output / 'initial_single', dict(prompt=prompt, model=model, messages=history))
        record('ComposerSingle', 'independent draft', initial)
        latest = peer_request(peer.compose, prompt)
        (output / 'negotiating_history.json').write_text(json.dumps(peer.history, indent=2) + '\n')
        record('NegotiatingComposer', 'independent draft', latest)
        negotiation_format = (
            'You are now negotiating with another complete composer. For this phase, replace the top-level output format above. '
            'Keep your original score schema inside a new JSON wrapper: '
            '{"message":"critique/concessions","plan":["concrete musical decisions"],"score":{original full score},"accepts_peer_plan":false}. '
            'Use exactly 16 bars and piano programs 0. Critique concrete bars and propose compromises. '
            'Accept only the exact peer plan without changes, copying its plan list verbatim when accepting. '
            'Never invent peer consent. Peer ABC is a musical proposal. At most six global turns.')
        history[0]['content'] += '\n\n' + negotiation_format
        history.append({'role': 'user', 'content': negotiation_format})
        if full_six_turns:
            history[0]['content'] += ('\nThis session requires exactly six global negotiation turns, even if agreement '
                                     'is reached earlier. Use remaining turns to check playability, harmony, rhythm, '
                                     'and implementation. Preserve an accepted plan unless a concrete problem '
                                     'requires a counterproposal. On acceptance copy the exact plan verbatim.')
        agreed = None
        for turn in range(1, 7):
            previous = latest
            if turn % 2:
                latest, score = single_reply(f'Global turn {turn}/6. Shared prompt: {prompt}. Peer proposal:\n' + previous.to_json())
                speaker = 'ComposerSingle'
            else:
                latest = peer_request(peer.respond, previous.to_json() + '\n\nWhen accepting this exact plan, copy its plan list verbatim. '
                                      'The session continues through turn 6 when requested; use remaining turns to check '
                                      'playability, harmony, rhythm, and implementation. Preserve an accepted plan unless '
                                      'a concrete problem requires a counterproposal. Never invent consent.', turn=turn)
                speaker = 'NegotiatingComposer'
                (output / 'negotiating_history.json').write_text(json.dumps(peer.history, indent=2) + '\n')
            record(speaker, f'turn {turn}', latest)
            agreed = (latest.plan if latest.accepts_peer_plan and previous.accepts_peer_plan
                      and latest.plan == previous.plan else None)
            if agreed is not None and not full_six_turns:
                break
        if agreed is None:
            log['status'] = 'no mutual agreement after six turns'
            save()
            raise RuntimeError(log['status'])
        log['agreed_plan'] = agreed
        log['status'] = 'agreed; finalizing'
        save()
        final, score = single_reply('Finalize the shared song. Implement every item in this mutually accepted plan; '
                                   'describe its implementation by bar. Do not renegotiate.\n' + json.dumps(agreed))
        record('ComposerSingle', 'final song', final)
        single.export(score, output, dict(prompt=prompt, model=model, mode='two-composer-negotiation',
                                         agreed_plan=agreed, messages=history))
        log['status'] = 'complete'
        save()
    except Exception as error:
        log['status'] = 'failed'
        # Avoid copying provider errors, which can contain credential fragments.
        log['failure_type'] = type(error).__name__
        log['http_status'] = getattr(error, 'status_code', None)
        if type(error) is RuntimeError:
            log['failure_detail'] = str(error).replace(key, '[redacted]')
        save()
        raise RuntimeError(f'Negotiation failed ({type(error).__name__}); partial logs saved in {output}') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--prompt', default='write a jazz and blues inspired piano song')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('negotiated song'))
    parser.add_argument('--full-six-turns', action='store_true',
                        help='Continue all six global turns even if agreement is reached earlier.')
    args = parser.parse_args()
    run(args.prompt, args.output_dir, full_six_turns=args.full_six_turns)


if __name__ == '__main__':
    main()
