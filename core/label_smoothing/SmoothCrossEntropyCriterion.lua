local SmoothCrossEntropyCriterion, Criterion = torch.class('nn.SmoothCrossEntropyCriterion', 'nn.Criterion')

function SmoothCrossEntropyCriterion:__init(weights, sizeAverage)
    Criterion.__init(self)
    self.lsm = nn.LogSoftMax()
    self.nll = nn.SmoothClassNLLCriterion(weights, sizeAverage)
