## Inspiration

One too many health insurance rejections. In America most of us pay a lot for our health insurance, but often get our claims denied on the grounds of "medical necessity", which is (in my opinion) frequently just an excuse for "I don't want to pay for this." This is especially common for folks with chronic or who fall outside of what is considered "normal."

Also as a trans person, and this being pride month and all of the laws restricting access to health care being passed, this was especially on my mind.

## What it does

Generate appeals to health insurance denials.


The appeals might not be super high quality, but if you've looked at any denials you've gotten from your health insurance company they probably are not very sensible either. We believe that a lot of insurance companies follow a "deny first" approach, and they are obligated to look at appeals so if we can lower the barrier to appeals we can increase the barrier on health insurance companies to issue denials.

## How we built it

We combined the output of multiple models (dolly 12b, biogpt, etc.) with CA's public insurance data to generate (low quality) syntehtic dataset for fine-tuning Falcon 7B on to generate health insurance appeals.


## Challenges we ran into

There were a lot of challenges, most of them coming from trying to use the wrong hardware or tool for the job -- but some data challenges too.

### Machines

Getting access to machines big enough to trains these models is kind of tricky. While I do have some Jetson AGXs, they don't support bitsandbytes so we couldn't use those tricks.

Thankfully the cloud takes credit cards!

#### Jetson AGX Notes:

Install torch from https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl (needs torch < 2 >=1.13) otherwise no GPU support.

If you get `module 'torch.distributed' has no attribute 'ReduceOp'` uninstall `deepspeed` since it does not play nicely with the NVIDIA torch fork. idk why but similar issues exist on OSX (see https://github.com/microsoft/DeepSpeed/issues/2830).

You (currently) can not train on the 32GB AGXs (not enough memory and bitsandbits does not work because ARM NEO libs are broken) but you can run inference.


### Data

Hard to get data, while I have a fair number of health insurance denials they are not super well organized.

I filed a FOIA requesting appeals data from the state, but for privacy reasons they could not share, and they did not have a redacted version of the data set available.

Thankfully they do have the appeals *decisions* publicly available, so we back convereted the appeals decisions into rejections and appeals and used this as a starting point. This was done using Dolly12b (although I do want to explore falcon 40b later).

See the dataset_tools directory for more.


## Accomplishments that we're proud of

We made a model! We found a creative solution to the (current) lack of data.

## What we learned

* bitsandbytes is *AWESOME* but has issues on NVIDIA ARM machines
* the cloud is a magical saving grace
* there are a lot of different libraries for fine tuning LLMs (we tried Dolly's built in trainer, FalconFineTune, and lit-parrot) and they're all what we could "works in progress.

## What's next for Fight Health Insurance

* Improving our training data (probably some mechanical turk, collecting more denials, etc.)
* Integrating the LLM into our website
* Getting the website deployed
* Integrating with a FAX API (health insurance companies *love* faxes)
* Getting an app made (maybe?)

# healthinsurance-llm
LLM for Generating Health Insurance Appeals

