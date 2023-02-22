FROM python:3.9-slim



WORKDIR /usr/app/src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY Train_BGFG_BCE_with_weightsUnet3.py ./

COPY Unet_LtS.py ./

COPY model_utils.py ./

COPY metrics_utils.py ./

COPY loss_utils.py ./

COPY data_utils.py ./

COPY configUnet3.py ./

CMD ["python", "./Train_BGFG_BCE_with_weightsUnet3.py"]
