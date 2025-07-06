# Tried several tools, this one works:
## patch cli.js
### Step 1: Download tool
https://www.npmjs.com/package/obfuscator-io-deobfuscator

### Step2: Run obfuscator
run: `obfuscator-io-deobfuscator cli_bak.js -o cli_3.js`

### Step3: Find login logic and bypass
Get the code of `cli_3.js`

find `function zw1` and references to `zw1`

find:
```js
if (D) {
  V.push({
    id: "oauth",
    component: f9.default.createElement(zw1, {
      onDone: Z
    })
  });
}
```

`D` is `let D = S_();`

And `S_` is:
```js
function S_() {
  let A = process.env.CLAUDE_CODE_USE_BEDROCK || process.env.CLAUDE_CODE_USE_VERTEX;
  let B = kQ().apiKeyHelper;
  let Q = process.env.ANTHROPIC_AUTH_TOKEN || B;
  let {
    source: D
  } = hJ(D91());
  return !(A || Q || D === "ANTHROPIC_API_KEY" || D === "apiKeyHelper");
}
```

So, make `S_` return `false` will bypass login wall:
```js
function S_(){
  return false;
}
```

Change cli.js and run `claude`, and we can see that the login was bypassed!

## set litellm proxy
### create litellm config.yaml
```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openrouter/openai/gpt-4o-mini
      api_base: https://openrouter.ai/api/v1
      api_key: <your_api_key>
  - model_name: claude-sonnet-4-20250514
    litellm_params:
      model: openrouter/anthropic/claude-sonnet-4
      api_base: https://openrouter.ai/api/v1
      api_key: <your_api_key>
  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: openrouter/anthropic/claude-3-5-haiku
      api_base: https://openrouter.ai/api/v1
      api_key: <your_api_key>
```
- `model`, make sure set with vendor prefix like `openrouter` or `bedrock`
- Add `claude-sonnet-4-20250514` and `claude-3-5-haiku-20241022` to your model_list

### Step 2:
Test litellm proxy
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-1234' -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "system",
        "content": "You are an LLM named gpt-4o"
      },
      {
        "role": "user",
        "content": "what is your name?"
      }
    ]
}'

curl -X POST 'http://0.0.0.0:4000/chat/completions' -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-1234' -d '{
    "model": "claude-3-5-haiku-20241022",
    "messages": [
      {
        "role": "user",
        "content": "what is your name?"
      }
    ]
}'

curl -L -X POST 'http://0.0.0.0:4000/v1/messages' \
-H 'content-type: application/json' \
-H 'x-api-key: $LITELLM_API_KEY' \
-H 'anthropic-version: 2023-06-01' \
-d '{
  "model": "anthropic-claude",
  "messages": [
    {
      "role": "user",
      "content": "Hello, can you tell me a short joke?"
    }
  ],
  "max_tokens": 100
}'
```


## set environment variables
### Step 1:
export ANTHROPIC_BASE_URL=http://0.0.0.0:4000/
export ANTHROPIC_API_KEY=<your_api_key>

### Step 2:
Run claude code by: `claude`
