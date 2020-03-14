# QF1: Laptop college1a

-------------------------------------------------------------------------------------------------
### Hoe werkt dit laptocollege?

Dit laptopcollege bevat meerdere opdrachten waarin de stof uit het QF1 college wordt uitgediept. Met behulp 
van *Python* binnen een *Jupyter notebook* ga je een aantal berekeningen uitvoeren. Deze berekeningen zijn er op gericht om inzicht te geven in quantummechanica en niet zozeer om Python te leren. De
meeste Python code die je nodig zal hebben wordt je dan ook aangeboden. Nieuw zal het gebruik zijn van Jupyter notebooks om de opdrachten te structureren en de *sympy* module in python om symbolische berekeningen uit te voeren.

-------------------------------------------------------------------------------------------------

### Python functies en modules

* **sympy**: symbolische rekenmodule voor python. https://www.sympy.org/en/index.html. De mogelijkheden zijn eindeloos, maar een paar handige functies:
   - integrate(f(x), (x, x0, x1)) integreert een functie f(x) van x0 tot x1. Als grens kan je simpel oneindig aangeven met **oo**
   - simplify(expression) maakt de uitdrukking (meestal) simpeler
* **sympy.physics.quantum**: quantum module binnen sympy https://docs.sympy.org/latest/modules/physics/quantum/index.html. Enkele functies die we gebruik?en tijdens het laptopcollege zijn:
   - Wavefunction(psi(x),x) om een golffunctie te definieren
   - Wavefunction.expr:  geeft de functie psi(x)
   - Wavefunction.conjugate(): geeft de complex geconjugeerde van de golffunctie
   - DifferentialOperator(g(f(x)),f(x)): definieert een quantummechanische operator
   - qapply(operator\*wavefunction): laat een operator los op een golffunctie
*  **ipywidgets**: python module voor het maken van interactieve graphics https://ipywidgets.readthedocs.io/en/latest/.
   - interactive(f(i), i=(i0,i1)): roept functie aan met 'slider bar' die waarden tussen i0 en i1 kan aannemen. Handig om bijvoorbeeld in f(i) een plaatje te maken.
* **matplotlib**: plotting in python die jullie al eerder hebben gebruikt https://matplotlib.org/. Maar nu ook te gebruiken voor het maken van animaties.
   - animation.FuncAnimation(): maak een animatie van bijvoorbeeld een tijdsafhankelijke golffunctie.
   


```python
import sympy
from sympy import *
from sympy.plotting import plot
from sympy import Derivative, Function, Symbol
from sympy.physics.quantum.operator import DifferentialOperator
from sympy.physics.quantum.state import Wavefunction
from sympy.physics.quantum.qapply import qapply

%matplotlib inline
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
```

---------------------------------------------------------------------------
## Opdracht 1: Een golfpakket

Gegeven $\Psi(x,t) = A \exp (-(x^2 + \jmath t))$ met $A$ en $a$ reeel en positief (Hint: we hebben $\hbar=1$ gezet).


```python
x,t=symbols('x, t',real=True)
Psi_no_normalization = Wavefunction(exp(-(x**2+ 1j*t)), x, (t,0,oo))
```

**1a)** Bepaal de normalisatieconstante A.


```python
A = 1/sqrt(integrate(Psi_no_normalization.conjugate().expr * Psi_no_normalization.expr, (x,-oo,+oo)))
print('De normalisatieconstante A=',A)
```

    De normalisatieconstante A= 2**(1/4)/pi**(1/4)
    

**1b)** Herdefinieer de golffunctie met de normalisatieconstante en verifieer je antwoord.


```python
Psi = Wavefunction(Psi_no_normalization.expr * A, x, (t,0,oo))
A = 1/sqrt(integrate(Psi.conjugate().expr * Psi.expr, (x,-oo,+oo)))
print('Nieuwe normalisatie =',A)

```

    Nieuwe normalisatie = 1
    

**1c)** Definieer de kinetische energie operator (*DifferentialOperator*) en laat deze werken op onze golffunctie met behulp van de *qapply()* functie. 


```python
# placeholder function voor de definitie van de kinetische energie operator in de volgende regel
f=Function('psi') 
# definitie vann de kinetische energie operator
E_kin = DifferentialOperator(-Derivative(f(x,t),(x,2))/2,f(x,t))
# laat de kinetische energie operator werken op de golffunctie
E_kin_Psi = simplify(qapply(E_kin*Psi))
# .... en laat maar even het resultaat zien
print("E_kin_operator*Psi =")
E_kin_Psi.expr
```

    E_kin_operator*Psi =
    




$\displaystyle \frac{\sqrt[4]{2} \left(1 - 2 x^{2}\right) e^{- 1.0 i t - x^{2}}}{\sqrt[4]{\pi}}$



**1d)** Definieer nu de 'andere kant' van de Schrodinger vergelijking: de $i \partial/\partial t$ kant dus. En laat deze operator ook op de golffunctie werken


```python
# definieer de tijdsafgeleide operator 
DT = DifferentialOperator(1j*Derivative(f(x,t),t),f(x,t))
# laat de operator werken op de golffunctie
DT_Psi = simplify(qapply(DT*Psi))
# .... en laat maar even het resultaat zien
DT_Psi.expr
simplify(DT_Psi.expr/Psi.expr)
```




$\displaystyle 1.0$



**1e)** Kan je uitvinden welke potentiele energie-operator hoort bij de $\Psi(x,t)$?


```python
V = Function('V')
V = simplify((DT_Psi.expr - E_kin_Psi.expr)/Psi.expr)
V
```




$\displaystyle 2 x^{2}$



--------------------------------------------------------------------------
## Opdracht 2: De oneindig diepe put

**2a)** Controleer dat de oplossingen voor de oneindig diepe put orthonormaal zijn voor de verschillende energie-niveaus $n$.


```python
# stap1: definieer de eigenfuncties
x = symbols('x',positive=True)
a = 1.0
def put_functie(n):
    w = Wavefunction(sin(n*pi*x/a)*sqrt(2/a),(x,0,a))
    return w

# definieer het inproduct
def inproduct(m,n):
    inproduct = integrate(put_functie(m).conjugate().expr*put_functie(n).expr,(x,0,a))
    return inproduct
```


```python
# print het inproduct uit voor verschillende combinaties van eigenfuncties
for i in range(1,4):
    for j in range(1,4):
        print("<",i,"|",j,">=",inproduct(i,j))
```

    < 1 | 1 >= 1.00000000000000
    < 1 | 2 >= 0
    < 1 | 3 >= 0
    < 2 | 1 >= 0
    < 2 | 2 >= 1.00000000000000
    < 2 | 3 >= 0
    < 3 | 1 >= 0
    < 3 | 2 >= 0
    < 3 | 3 >= 1.00000000000000
    

**2b)** Maak een grafiek van de oplossingen voor een oneindig diepe put voor $a=1$ met $n$ als het te kiezen energie-niveau.


```python
#
# definieer een plot functie voor put_functie_n(x)
#


def wave_plot(n):
    #
    # converteer de sympy expressie in een numerieke functie om met matplotlib te gebruiken
    #
    put_l = lambdify(x,put_functie(n).expr, 'numpy')
    #
    # bereken de x en t waarden om te plotten
    #
    xx = np.arange(0.,a,0.01)
    yy = put_l(xx)
    #
    # en plotten maar.....
    #
    plt.plot(xx,yy,'r-')
    plt.ylim([-2.,2.])
    plt.xlabel('x')
    ytitle = '$\psi_{'+str(n)+'}(x)$'
    plt.ylabel(ytitle)
    plt.axhline(color='black')
    plt.show()
    #my_plot = plot(put_functie(n).expr,(x,0.,a))
#
# ... en maak hier de interactieve plot
#
interactive_plot = interactive(wave_plot, n=(1, 10), ylim=([-2.,2.]))
interactive_plot
```


    interactive(children=(IntSlider(value=5, description='n', max=10, min=1), Output()), _dom_classes=('widget-int…



```python
!ls

```

    'ls' is not recognized as an internal or external command,
    operable program or batch file.
    


```python
!nbconvert.exe
```

    'nbconvert.exe' is not recognized as an internal or external command,
    operable program or batch file.
    


```python

```
