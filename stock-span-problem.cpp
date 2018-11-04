// Link= https://practice.geeksforgeeks.org/problems/stock-span-problem/0
#include <bits/stdc++.h>
#define ll long long int
#define f(in,lt,st) for(int i=in;i<lt;i+=st)
using namespace std;

void input(int a[],int len);

int main(){
    long long int test;
    cin>>test;
    while(test--){
        int n;
        cin>>n;
        int a[n];
        input(a,n);
        stack<int> s;
        s.push(0);
        cout<<"1 ";
        for(int i=1;i<n;i++){
            if(s.empty()){
                cout<<i+1<<" ";
                s.push(i);
            }
            else{
                while(!s.empty()&& a[s.top()]<=a[i]){
                    s.pop();
                }
                if(s.empty()){
                    cout<<i+1<<" ";
                    s.push(i);
                }

                else{
                    cout<<i-s.top()<<" ";
                    s.push(i);
                }
            }
        }
        cout<<endl;
    }
    return 0;
}

void input(int a[],int len){
    for(ll i=0;i<len;i++)
        cin>>a[i];
}