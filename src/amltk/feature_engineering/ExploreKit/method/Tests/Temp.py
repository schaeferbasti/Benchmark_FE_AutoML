
from Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton

def main():
    oaas = OperationAssignmentAncestorsSingleton()
    oaas2 = OperationAssignmentAncestorsSingleton()

    oaas.addAssignment('add','x','y')
    oaas2.addAssignment('sub', 'x','y')

    print(oaas2.getSources('add'))
    print(oaas.getSources('sub'))
    print(oaas.getTargets('sub'))
    print(oaas.getTargets('asd'))

if __name__ == '__main__':
    main()