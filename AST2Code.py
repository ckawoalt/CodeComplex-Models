import javalang
import random

class AST2Code_module():
    def __init__(self,dead_code=0):
        r'''
        init module for ast -> code
        Arguments:
        
        probs (:obj:`dict`):
            probability of code augmnetation
            dictionary contaian each augmantaion probability

            key value must be ['for','ternary','method','useless','variable']
        '''
        #function dictornary for node-function matching.key: node name , value : node convert function
        self.FUNC_DICT={
        "ClassCreator":self.ClassCreator2Code,
        "Literal":self.Literal2Code,
        'BinaryOperation':self.BinaryOperation2Code,
        'MemberReference':self.MemberReference2Code,
        'BlockStatement':self.BlockStatement2Code,
        'list':self.List2Code,
        'LocalVariableDeclaration':self.LocalVariableDeclaration2Code, #variavle Declaration
        'VariableDeclaration':self.VariableDeclaration2Code,
        'VariableDeclarator':self.VariableDeclarator2Code,
        'ArrayCreator': self.ArrayCreator2Code,
        'StatementExpression':self.StatementExpression2Code,
        'ArraySelector':self.ArraySelector2Code,
        'MethodInvocation':self.MethodInvocation2Code,
        'Assignment':self.Assignment2Code,
        'IfStatement':self.IfStatement2Code,
        'ForStatement':self.ForStatement2Code,
        'MethodDeclaration':self.MethodDeclaration2Code,
        'Cast':self.Cast2Code,
        'ReturnStatement':self.ReturnStatement2Code,
        'ClassDeclaration':self.ClassDeclaration2Code,
        'FieldDeclaration':self.FieldDeclaration2Code,
        'NoneType':self.Nonetype,
        'ConstructorDeclaration':self.ConstructorDeclaration2Code,
        'WhileStatement':self.WhileStatement2Code,
        'TryStatement':self.TryStatement2Code,
        'CatchClauseParameter':self.CatchClauseParameter2Code,
        'ThrowStatement':self.ThrowStatement2Code,
        'CatchClause':self.CatchClause2Code,
        'TryResource':self.TryResource2Code,
        'This':self.This2Code,
        'TernaryExpression':self.TernaryExpression2Code,
        'BreakStatement':self.BreakStatement2Code,
        'ExplicitConstructorInvocation':self.ExplicitConstructorInvocation2Code,
        'DoStatement':self.DoStatement2Code,
        'InterfaceDeclaration':self.InterfaceDeclaration2Code,
        'ContinueStatement':self.ContinueStatement2Code,
        'Statement':self.Statement2Code,
        'ArrayInitializer':self.ArrayInitializer2Code,
        'SuperConstructorInvocation':self.SuperConstructorInvocation2Code,
        'SwitchStatement':self.SwitchStatement2Code,
        'EnhancedForControl':self.EnhancedForControl2Code,
        'ForControl':self.ForControl2Code,
        'ReferenceType':self.Type2Code,
        'AssertStatement':self.AssertStatement2Code,
        'SuperMethodInvocation':self.SuperMethodInvocation2Code,
        'BasicType':self.BasicType2Code,
        'LambdaExpression':self.LambdaExpression2Code,
        'FormalParameter':self.FormalParameter2Code,
        'InferredFormalParameter':self.InferredFormalParameter2Code,
        'MethodReference':self.MethodReference2Code,
        'ConstantDeclaration':self.ConstantDeclaration2Code,
        'EnumDeclaration':self.EnumDeclaration2Code,
        'EnumBody':self.EnumBody2Code,
        'EnumConstantDeclaration':self.EnumConstantDeclaration2Code,
        'InnerClassCreator':self.InnerClassCreator2Code
        }

        self.dead_code=dead_code#for deadcode elimiation probability
    
    def AST2Code(self,root):
        code=''
        self.check_dead_code(root)
        #start convert
        if root.package:
            code+= 'package '+root.package.name+';\n'
        for import_item in root.imports:
            code+= 'import '+('static ' if import_item.static else '') +import_item.path+ ('.*' if import_item.wildcard else '') + ';\n'

        for child in root.types:
            code+=self.FUNC_DICT[type(child).__name__](node=child)
        return code

    def check_dead_code(self,tree):
        self.valid_method=set()
        self.valid_class=set()
        self.valid_method.add('main')
        for _,node in tree:
            if type(node).__name__ == 'ClassDeclaration':
                for class_body in node.body:
                    if type(class_body).__name__ == 'MethodDeclaration' and class_body.name=='main':
                        self.valid_class.add(node.name)
            elif type(node).__name__=='MethodInvocation':
                self.valid_method.add(node.member)
                
            elif type(node).__name__=='ClassCreator':
                self.valid_class.add(self.Type2Code(node=node.type))

    def split_method(self,root):
        '''
        for new model. code split to class,method
        '''
        code_array=[]
        self.prepare_convert(root)
        for node in root.types:
            if type(node).__name__=='ClassDeclaration': # find class 
                method_array=[]
                for node_child in node.body:
                    if type(node_child).__name__=='MethodDeclaration' and node_child.name =='main':
                        method_index=len(method_array)
                        cla_index=len(code_array)
                    method_array.append(self.FUNC_DICT[type(node_child).__name__](node=node_child))
                if len(method_array)>0:
                    code_array.append(method_array)
        return code_array,{"method":method_index,"class":cla_index}

    def Nonetype(self,**kwargs):
        return ''

    def FieldDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        
        modifier_arr=[]
        for modifier in node.modifiers:
            modifier_arr.append(modifier)
        code_snippet+=' '.join(reversed(modifier_arr)) +' '
        
        code_snippet+=self.Type2Code(node=node.type)+' '
        for declarator in node.declarators:
            code_snippet+= self.FUNC_DICT[type(declarator).__name__](node=declarator)+','
        code_snippet=code_snippet.rstrip(',')

        return code_snippet+';\n'

    def WhileStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.label:
            code_snippet+=node.label+':'

        code_snippet+='while('+self.FUNC_DICT[type(node.condition).__name__](node=node.condition,inline=False)+')'
        code_snippet+=self.FUNC_DICT[type(node.body).__name__](node=node.body)
        return code_snippet

    def ConstructorDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        code_snippet+=' '.join(reversed(list(node.modifiers)))
        code_snippet+=' '+ node.name
        code_snippet=code_snippet.lstrip()
        paramter=''
    
        for pram in node.parameters:
            paramter+=' '.join(reversed(list(pram.modifiers))) +' '
            paramter+=self.Type2Code(node=pram.type)
            paramter+=('...' if pram.varargs else '')
            paramter+=' '+ pram.name +','

        code_snippet+='('+paramter.rstrip(',')+')'
        code_snippet+= ('throws ' +','.join(node.throws) if node.throws else '')
        code_snippet+='{\n'
        for child in node.body:
            code_snippet+= self.FUNC_DICT[type(child).__name__](node=child)
        return code_snippet+'}\n'

    def ClassDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        if node.name in self.valid_class or random.random() > self.probs['unreachble']: #reverse condition
            code_snippet=''
            for annot in node.annotations:
                code_snippet+='@'+annot.name+'('+self.FUNC_DICT[type(annot.element).__name__](node=annot.element)+')\n'
            code_snippet+=(node.documentation+'\n' if node.documentation else '')
            modifier_arr=[]
            for modifier in node.modifiers:
                modifier_arr.append(modifier)
            code_snippet+=' '.join(reversed(modifier_arr)) +' '
    
            code_snippet+=' class '+node.name
            if node.type_parameters:
                code_snippet+='<'
                for param in node.type_parameters:
                    code_snippet+=param.name+','
                code_snippet=code_snippet.rstrip(',') +'>'

            if node.extends:
                code_snippet+=' extends '+self.Type2Code(node=node.extends)
            if node.implements:
                code_snippet+=' implements '
                code_snippet+=','.join(self.Type2Code(node=implement)for implement in node.implements)
            code_snippet+='{\n'
            for class_item in node.body:
                code_snippet+=self.FUNC_DICT[type(class_item).__name__](node=class_item)
            code_snippet+='}\n'
        else:
            code_snippet=''
        return code_snippet

    def MethodDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        
        if node.name in self.valid_method or random.random() > self.dead_code:
            code_snippet=(node.documentation+'\n' if node.documentation else '')
            if node.annotations:
                for annotation in node.annotations:
                    code_snippet+='@'+annotation.name+'\n'
            code_snippet+=' '.join(reversed(list(node.modifiers))) +' '

            
            if node.type_parameters:
                code_snippet+='<'
                for type_param in node.type_parameters:
                    code_snippet+=type_param.name+','
                code_snippet=code_snippet.rstrip(',')+'> '
                
            code_snippet+=(self.Type2Code(node=node.return_type) if node.return_type else 'void')+' '+node.name+'('
            paramter=''
            
            for pram in node.parameters:
                paramter+=' '.join(reversed(list(pram.modifiers))) +' '
                paramter+=self.Type2Code(node=pram.type)
                paramter+=('...' if pram.varargs else '')
                paramter+=' '+ pram.name+','

            code_snippet+= paramter.rstrip(',')+')'+('throws ' +','.join(node.throws)+' ' if node.throws else '')
            code_snippet += '{\n'
            if node.body:
                for method_item in node.body:
                    code_snippet+=self.FUNC_DICT[type(method_item).__name__](node=method_item)
            code_snippet+= '}\n'
        else:
            code_snippet=''
        return code_snippet
  

    def InterfaceDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=(node.documentation+'\n' if node.documentation else '')
        if node.annotations:
            for annotation in node.annotations:
                code_snippet+='@'+annotation.name+'\n'
        for modifier in node.modifiers:
            code_snippet+=modifier+' ' 

        code_snippet+=' interface '+node.name
        if node.type_parameters:
            code_snippet+='<'
            code_snippet+=','.join(pram.name for pram in node.type_parameters)
            code_snippet+='>'
            
        if node.extends:
            code_snippet+=' extends '+','.join(self.Type2Code(node=extend) for extend in node.extends)
        
        code_snippet+='{\n'
        for b in node.body:
            if type(b).__name__=='MethodDeclaration' and b.body==None:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b).rstrip('{\n}\n')+';\n'
            else:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
        code_snippet+='}\n'
        return code_snippet

    def ConstantDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=(node.documentation+'\n' if node.documentation else '')
        for annot in node.annotations:
            code_snippet+='@'+annot.name+'('+self.FUNC_DICT[type(annot.element).__name__](node=annot.element)+')\n'
        modifier_arr=[]
        for modifier in node.modifiers:
            modifier_arr.append(modifier)
        code_snippet+=' '.join(reversed(modifier_arr)) +' '
        code_snippet+=self.Type2Code(node=node.type)
        code_snippet += ' ' +','.join(self.FUNC_DICT[type(declarator).__name__](node=declarator)  for declarator in node.declarators)

        return code_snippet+';\n'

    def LocalVariableDeclaration2Code(self,**kwargs):
        node=kwargs['node']
 
        code_snippet=' '.join(reversed(list(node.modifiers))) +' '
        code_snippet+=self.Type2Code(node=node.type)+' '
        for declarator in node.declarators:
            code_snippet += self.FUNC_DICT[type(declarator).__name__](node=declarator) +','

        code_snippet=code_snippet.rstrip(',')
        return code_snippet+'; \n'

    def VariableDeclarator2Code(self,**kwargs):
        node=kwargs['node']
        
        name_modifier=''
        code_snippet=''

        '''
        if node.qualifier in self.object_variable:
            code_snippet+=name_modifier+self.variable_dict[node.name]
        else:    
            code_snippet+=node.name # normal option
        '''
        code_snippet+=name_modifier+(node.name if node.name else '')
        #argmentation option
        code_snippet+=('[]'*len(node.dimensions) if node.dimensions else '')+' ' 
     
        if node.initializer:
            code_snippet+= ' = '+self.FUNC_DICT[type(node.initializer).__name__](node=node.initializer)

        return code_snippet

    def VariableDeclaration2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet=''
        code_snippet+=self.Type2Code(node=node.type)+' '
        operandr=''
        operandl=''

        for declarator in node.declarators:
            operandl += self.FUNC_DICT[type(declarator).__name__](node=declarator)+','
        
        code_snippet +=operandr.rstrip(',')+operandl.rstrip(',')
        return code_snippet
        
    def Type2Code(self,**kwargs):
        node=kwargs['node']

        if type(node).__name__ == 'BasicType':
            return node.name +('[]'*len(node.dimensions) if node.dimensions else '')
        elif type(node).__name__ == 'ReferenceType':
            code_snippet=node.name
            if node.arguments != None:
                code_snippet+='<'
                for arg in node.arguments:
                    if arg.pattern_type:
                        if arg.pattern_type=='extends':
                            code_snippet+='? ' +arg.pattern_type+' '
                        else:
                            code_snippet+=arg.pattern_type+','
                    if arg.type:
                        code_snippet+=self.Type2Code(node=arg.type)+','
                code_snippet=code_snippet.rstrip(',')+'>'
            
            if node.sub_type:
                code_snippet+='.'+self.Type2Code(node=node.sub_type)
            code_snippet+=('[]'*len(node.dimensions) if node.dimensions else '')

            return code_snippet

    def ClassCreator2Code(self,**kwargs):
        node=kwargs['node']
        
        argments='('
        for arg in node.arguments:
            argments+=self.FUNC_DICT[type(arg).__name__](node=arg)+','
        
        argments=argments.rstrip(',')+')'
        code_snippet='new '+self.Type2Code(node=node.type)+argments
        if node.body:
            code_snippet+='{'
            for b in node.body:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
            code_snippet+='}'
        if node.selectors:
            for index in node.selectors:
                if type(index).__name__ != 'ArraySelector':
                    code_snippet+='.'    
                code_snippet+=self.FUNC_DICT[type(index).__name__](node=index,inline=False)    
        return code_snippet

    def MemberReference2Code(self,**kwargs):
        node=kwargs['node']
        
        code_snippet=(''.join(node.prefix_operators) if node.prefix_operators else '')+(node.qualifier+'.' if node.qualifier else '')
        code_snippet+=node.member 
        selector=''
        if node.selectors:
            for index in node.selectors:
                if type(index).__name__ != 'ArraySelector':
                    selector+='.'    
                selector+=self.FUNC_DICT[type(index).__name__](node=index)
        code_snippet+=selector
        return code_snippet+(''.join(node.postfix_operators) if node.postfix_operators else '')
    def MethodReference2Code(self,**kwargs):
        r"""
        Methodreference type to code snippet

        Arguments:
            node (:obj:`javalang.node.tree.MethodReference`):
                target node
            
        Returns:
            :Boolean : if match condtions return True else False 
        """
    
        node=kwargs['node']
        code_snippet=''
        
        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        code_snippet+='::'
        code_snippet+=self.FUNC_DICT[type(node.method).__name__](node=node.method)
        

        return code_snippet
    def ForStatement2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet='for ( '
        code_snippet+=self.FUNC_DICT[type(node.control).__name__](node=node.control)
        code_snippet+= ')\n'
        
        code_snippet+= self.FUNC_DICT[type(node.body).__name__](node=node.body)

        return code_snippet
    def ForControl2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.init:
            if type(node.init) == list:
                code_snippet+= ','.join([self.FUNC_DICT[type(i).__name__](node=i).rstrip(';\n') for i in node.init])
            else:
                code_snippet+=self.FUNC_DICT[type(node.init).__name__](node=node.init).rstrip(';\n')
        
        code_snippet+=';'
        if node.condition:
            code_snippet+=self.FUNC_DICT[type(node.condition).__name__](node=node.condition)
        code_snippet +=';'
        if node.update:
            
            code_snippet+= ','.join([self.FUNC_DICT[type(update).__name__](node=update).rstrip(';\n') for update in node.update])
        return code_snippet

    def EnhancedForControl2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        code_snippet+=self.FUNC_DICT[type(node.var).__name__](node=node.var)+':'
        code_snippet+=self.FUNC_DICT[type(node.iterable).__name__](node=node.iterable)
        return code_snippet

    def IfStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='if ( '
        code_snippet+= self.FUNC_DICT[type(node.condition).__name__](node=node.condition)+')'
        code_snippet+='\n'+self.FUNC_DICT[type(node.then_statement).__name__](node=node.then_statement)+'\n'
        if node.else_statement:
            code_snippet+='else '+self.FUNC_DICT[type(node.else_statement).__name__](node=node.else_statement)
        return code_snippet
        
    def BlockStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet="{"
        for block_item in node.statements:
            code_snippet+=self.FUNC_DICT[type(block_item).__name__](node=block_item)
        code_snippet+="}"
        return code_snippet

    def List2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet="{"
        for list_item in node:
            code_snippet+=self.FUNC_DICT[type(list_item).__name__](node=list_item)
        code_snippet+="}"
        return code_snippet
        
    def BinaryOperation2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        operandl=self.FUNC_DICT[type(node.operandl).__name__](node=node.operandl,replace=type(node.operandl).__name__ == 'MemberReference')
        if type(node.operandl).__name__== 'Assignment':
            code_snippet+='('+operandl+')'
        else:
            code_snippet+= operandl
        code_snippet+=' ' +node.operator +' '
        
        operandr=self.FUNC_DICT[type(node.operandr).__name__](node=node.operandr,replace=type(node.operandr).__name__ == 'MemberReference')
        if type(node.operandr).__name__== 'Assignment':
            code_snippet+='('+operandr+')'
        else:
            code_snippet+= operandr
        return '('+code_snippet+')'

    def Literal2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=node.value
        prefix=''.join(node.prefix_operators if node.prefix_operators else '')
        postfix=''.join(node.postfix_operators if node.postfix_operators else '')
        for s in node.selectors:
            if not type(s).__name__ =='ArraySelector':
                code_snippet+= '.'    
            code_snippet+=self.FUNC_DICT[type(s).__name__](node=s)
        return prefix+code_snippet+postfix

    def ArrayCreator2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='new '
        code_snippet+= self.Type2Code(node=node.type)
        for dim in node.dimensions:
            code_snippet+= '['+self.FUNC_DICT[type(dim).__name__](node=dim)+']'
        if node.initializer:
            code_snippet+=self.FUNC_DICT[type(node.initializer).__name__](node=node.initializer)
        return code_snippet
 
    def ArrayInitializer2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='{'
        for init in node.initializers:
            code_snippet+=self.FUNC_DICT[type(init).__name__](node=init)+','
        return code_snippet.rstrip(',')+'}'

    def Cast2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='('+self.Type2Code(node=node.type)+')'

        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        return code_snippet

    def StatementExpression2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression,end=';\n')

        return code_snippet+';\n'

    def Assignment2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet=self.FUNC_DICT[type(node.expressionl).__name__](node=node.expressionl)

        code_snippet+=' ' +node.type+' '
        code_snippet+= self.FUNC_DICT[type(node.value).__name__](node=node.value)
        return code_snippet

    def MethodInvocation2Code(self,**kwargs):
        
        node=kwargs['node']
        args=[self.FUNC_DICT[type(arg).__name__](node=arg) for arg in node.arguments]
        args=','.join(args)
        code_snippet=''

        code_snippet+=(node.qualifier+'.' if node.qualifier else '')
        code_snippet+= node.member+'('
        code_snippet+=args+')'

        if node.selectors:
            for s in node.selectors:
                if not type(s).__name__ in ['ArraySelector']:
                    code_snippet+= '.'
                code_snippet+=self.FUNC_DICT[type(s).__name__](node=s)
        prefix_operators=node.prefix_operators if node.prefix_operators else []
        postfix_operators=node.postfix_operators if node.postfix_operators else []
        
        return ''.join(prefix_operators)+code_snippet+''.join(postfix_operators)

    def ArraySelector2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='['
        code_snippet+=self.FUNC_DICT[type(node.index).__name__](node=node.index)
        code_snippet+=']'
        
        return code_snippet

    def ReturnStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='return '
        expression=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        
        return code_snippet+expression+';'

    def TryStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='try'
        if node.resources:
            code_snippet+='('
            resource_code=[]
            for resource in node.resources:
                resource_code.append(self.FUNC_DICT[type(resource).__name__](node=resource))
            code_snippet+=';'.join(resource_code)+')'
            
        code_snippet+='{'
        for b in node.block:
            code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
        code_snippet+='\n}'
        if node.catches:
            for catch in node.catches:
                code_snippet+= self.FUNC_DICT[type(catch).__name__](node=catch)
        if node.finally_block:
            code_snippet+='finally{'
            for b in node.finally_block:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
            code_snippet+='\n}'
        return code_snippet

    def CatchClause2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet= 'catch ('+self.FUNC_DICT[type(node.parameter).__name__](node=node.parameter) +')'
        code_snippet+='{\n'
        for catch_block in node.block:
            code_snippet+=self.FUNC_DICT[type(catch_block).__name__](node=catch_block)
        code_snippet+='}\n'

        return code_snippet

    def CatchClauseParameter2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        for p in node.types:
            code_snippet+=p+' '
        code_snippet+=node.name
        return code_snippet

    def ThrowStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='throw ('
        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        code_snippet+=')'
        return code_snippet +';'

    def TryResource2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=self.Type2Code(node=node.type)+' '
        code_snippet+= node.name

        code_snippet+='='+self.FUNC_DICT[type(node.value).__name__](node=node.value)
        return code_snippet

    def This2Code(self,**kwargs):
        node=kwargs['node']
        prefix=''.join(node.prefix_operators if node.prefix_operators else '')
        postfix=''.join(node.postfix_operators if node.postfix_operators else '')

        code_snippet='this'
        for s in node.selectors:
            if not type(s).__name__ =='ArraySelector':
                code_snippet+= '.'    
            code_snippet+=self.FUNC_DICT[type(s).__name__](node=s)
        return prefix+code_snippet+postfix

    def TernaryExpression2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet=''    
        code_snippet+='('
        if node.condition:
            if type(node.condition).__name__ ==' MethodInvocation':
                code_snippet=self.FUNC_DICT[type(node.condition).__name__](node=node.condition,method_target=code_snippet)
            else:
                code_snippet+=self.FUNC_DICT[type(node.condition).__name__](node=node.condition)
        code_snippet+='?'
        if node.if_true:
            if type(node.if_true).__name__ == 'Assignment':
                code_snippet+='('+self.FUNC_DICT[type(node.if_true).__name__](node=node.if_true,inline=False) +')'
            else:
                code_snippet+=self.FUNC_DICT[type(node.if_true).__name__](node=node.if_true,inline=False)
        code_snippet+=':'
        if node.if_false:
            if type(node.if_false).__name__ == 'Assignment':
                code_snippet+='('+self.FUNC_DICT[type(node.if_false).__name__](node=node.if_false,inline=False) +')'
            else:
                code_snippet+=self.FUNC_DICT[type(node.if_false).__name__](node=node.if_false,inline=False)
        code_snippet+=')'

        return code_snippet

    def BreakStatement2Code(self,**kwargs):

        code_snippet='break'
        return code_snippet+';'

    def ExplicitConstructorInvocation2Code(self,**kwargs):
        node=kwargs['node']
        
        code_snippet='this('
        for argument in node.arguments:
            code_snippet+=self.FUNC_DICT[type(argument).__name__](node=argument)+','
        code_snippet=code_snippet.rstrip(',')
        code_snippet+=')'
        return code_snippet

    def DoStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='do '+self.FUNC_DICT[type(node.body).__name__](node=node.body)
        code_snippet+='while('+self.FUNC_DICT[type(node.condition).__name__](node=node.condition,inline=False)+')'
        return code_snippet+';'

    def ContinueStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='continue' 
        if node.goto:
            code_snippet+=' '+node.goto
        return code_snippet+';'

    def Statement2Code(self,**kwargs):

        return ';'

    def SuperConstructorInvocation2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='super'
        code_snippet+='('
        if node.arguments:
            code_snippet+=','.join([self.FUNC_DICT[type(arg).__name__](node=arg) for arg in node.arguments])

        code_snippet+=')'
            
        return code_snippet

    def SwitchStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='switch('+self.FUNC_DICT[type(node.expression).__name__](node=node.expression,inline=False)+'){\n'

        for case in node.cases:
            if case.case:
                code_snippet+='case '
                for cons in case.case:
                    code_snippet+=self.FUNC_DICT[type(cons).__name__](node=cons)+','
                code_snippet=code_snippet.rstrip(',')
            else:
                code_snippet+='default '
            code_snippet+=':'
            for statement in case.statements:
                code_snippet+=self.FUNC_DICT[type(statement).__name__](node=statement)
            code_snippet+='\n'

        return code_snippet+'}\n'

    def AssertStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='assert '
        code_snippet+=self.FUNC_DICT[type(node.condition).__name__](node=node.condition)
        return code_snippet+';\n'

    def SuperMethodInvocation2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='super.'
        code_snippet+=node.member
        return code_snippet+'()'

    def BasicType2Code(self,**kwargs):
        node=kwargs['node']
    
        return node.name +('[]'*len(node.dimensions) if node.dimensions else '')

    def LambdaExpression2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        code_snippet+=','.join(self.FUNC_DICT[type(pram).__name__](node=pram) for pram in node.parameters)
        code_snippet='('+code_snippet+')'

        code_snippet+='->'
        if type(node.body) ==list:
                code_snippet+='{'
                code_snippet+=''.join(self.FUNC_DICT[type(b).__name__](node=b,inline=False) for b in node.body)
                code_snippet+='}'
        else:
            code_snippet+=self.FUNC_DICT[type(node.body).__name__](node=node.body)
        return code_snippet

    def FormalParameter2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=self.Type2Code(node=node.type)+' '+node.name
        return code_snippet

    def InferredFormalParameter2Code(self,**kwargs):
        node=kwargs['node']
        return node.name

    def EnumDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='enum '+node.name+'{'
        code_snippet+=self.FUNC_DICT[type(node.body).__name__](node=node.body)
        code_snippet+='}'
        return code_snippet

    def EnumBody2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        constants_items=[self.FUNC_DICT[type(constant).__name__](node=constant) for constant in node.constants]
        code_snippet+=','.join(constants_items)
        return code_snippet

    def EnumConstantDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=node.name
        return code_snippet

    def InnerClassCreator2Code(self,**kwargs):
        node=kwargs['node']
        
        argments='('
        for arg in node.arguments:
            argments+=self.FUNC_DICT[type(arg).__name__](node=arg)+','
        
        argments=argments.rstrip(',')+')'
        code_snippet='new '+self.Type2Code(node=node.type)+argments
        if node.body:
            code_snippet+='{'
            for b in node.body:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
            code_snippet+='}'
        if node.selectors:
            for index in node.selectors:
                if type(index).__name__ != 'ArraySelector':
                    code_snippet+='.'    
                code_snippet+=self.FUNC_DICT[type(index).__name__](node=index,inline=False)    
        return code_snippet