# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Note: huggingface_hub==0.26.5 fails as in https://github.com/easydiffusion/easydiffusion/issues/1851
RUN pip install hydra-core torchinfo blobfile
RUN pip install mpi4py
# RUN pip install --no-dependencies torchaudio==2.6.0
# RUN git clone https://github.com/pytorch/audio
# RUN cd audio; pip install -v -e . --no-use-pep517
RUN pip install pyloudnorm
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
RUN pip install transformers==4.52.3

WORKDIR /workspace

RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]