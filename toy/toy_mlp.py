resnet = torchvision.models.resnet50(pretrained=True)
avgpool = nn.Sequential(list(resnet.children())[-2])

avgpool should be in the init