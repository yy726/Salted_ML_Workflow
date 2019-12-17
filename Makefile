.PHONY: data

data:
	aws s3 cp s3://2019fa-facedetection/pretrained_models pretrained_models --request-payer=requester