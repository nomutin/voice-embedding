# voice-embedding

## Deploy

```bash
make deploy
```

## Invoke

```bash
curl -X POST -H "Content-Type:application/octet-stream" --data-binary @tests/sample.wav https://xxxxxxxx.lambda-url.<aws-region>.on.aws/
```
