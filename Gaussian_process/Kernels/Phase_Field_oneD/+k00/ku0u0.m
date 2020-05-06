function [K] = ku0u0(x, xp, hyp, ubar, ubarp, dt, i)

logsigma = hyp(1);
logtheta = hyp(2);

a1 = hyp(3);
a2 = hyp(4);
a3 = hyp(5);
a4 = hyp(6);

n_x = size(x,1);
n_xp = size(xp,1);

x = repmat(x,1,n_xp);
xp = repmat(xp',n_x,1);

ubar = repmat(ubar,1,n_xp);
ubarp = repmat(ubarp',n_x,1);

switch i


case 0

K=(1/4).*a4.^(-2).*exp(1).^(logsigma+(-4).*logtheta+(-1/2).*exp(1).^((-1) ...
  .*logtheta).*(x+(-1).*xp).^2).*(exp(1).^(4.*logtheta).*(4.*a4.^2+dt.^2.* ...
  ((-1)+ubar).*(2.*a3+a2.*((-1)+2.*ubar)).*((-1)+ubarp).*(2.*a3+a2.*((-1)+ ...
  2.*ubarp))+2.*a4.*dt.*(2.*a3.*((-2)+ubar+ubarp)+a2.*(2+(-3).*ubar+2.* ...
  ubar.^2+(-3).*ubarp+2.*ubarp.^2)))+2.*a1.*dt.*exp(1).^(2.*logtheta).*( ...
  4.*a4+2.*a3.*dt.*((-2)+ubar+ubarp)+a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3) ...
  .*ubarp+2.*ubarp.^2)).*(exp(1).^logtheta+(-1).*(x+(-1).*xp).^2)+4.* ...
  a1.^2.*dt.^2.*(3.*exp(1).^(2.*logtheta)+(-6).*exp(1).^logtheta.*(x+(-1) ...
  .*xp).^2+(x+(-1).*xp).^4));


case 1 % logsigma

K=(1/4).*a4.^(-2).*exp(1).^(logsigma+(-4).*logtheta+(-1/2).*exp(1).^((-1) ...
  .*logtheta).*(x+(-1).*xp).^2).*(exp(1).^(4.*logtheta).*(4.*a4.^2+dt.^2.* ...
  ((-1)+ubar).*(2.*a3+a2.*((-1)+2.*ubar)).*((-1)+ubarp).*(2.*a3+a2.*((-1)+ ...
  2.*ubarp))+2.*a4.*dt.*(2.*a3.*((-2)+ubar+ubarp)+a2.*(2+(-3).*ubar+2.* ...
  ubar.^2+(-3).*ubarp+2.*ubarp.^2)))+2.*a1.*dt.*exp(1).^(2.*logtheta).*( ...
  4.*a4+2.*a3.*dt.*((-2)+ubar+ubarp)+a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3) ...
  .*ubarp+2.*ubarp.^2)).*(exp(1).^logtheta+(-1).*(x+(-1).*xp).^2)+4.* ...
  a1.^2.*dt.^2.*(3.*exp(1).^(2.*logtheta)+(-6).*exp(1).^logtheta.*(x+(-1) ...
  .*xp).^2+(x+(-1).*xp).^4));


case 2 % logtheta

K=(1/4).*a4.^(-2).*exp(1).^(logsigma+(-4).*logtheta+(-1/2).*exp(1).^((-1) ...
  .*logtheta).*(x+(-1).*xp).^2).*(2.*a1.*dt.*exp(1).^(3.*logtheta).*(4.* ...
  a4+2.*a3.*dt.*((-2)+ubar+ubarp)+a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3).* ...
  ubarp+2.*ubarp.^2))+4.*exp(1).^(4.*logtheta).*(4.*a4.^2+dt.^2.*((-1)+ ...
  ubar).*(2.*a3+a2.*((-1)+2.*ubar)).*((-1)+ubarp).*(2.*a3+a2.*((-1)+2.* ...
  ubarp))+2.*a4.*dt.*(2.*a3.*((-2)+ubar+ubarp)+a2.*(2+(-3).*ubar+2.* ...
  ubar.^2+(-3).*ubarp+2.*ubarp.^2)))+24.*a1.^2.*dt.^2.*exp(1).^logtheta.*( ...
  exp(1).^logtheta+(-1).*(x+(-1).*xp).^2)+4.*a1.*dt.*exp(1).^(2.*logtheta) ...
  .*(4.*a4+2.*a3.*dt.*((-2)+ubar+ubarp)+a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+( ...
  -3).*ubarp+2.*ubarp.^2)).*(exp(1).^logtheta+(-1).*(x+(-1).*xp).^2)+(exp( ...
  1).^(4.*logtheta).*(4.*a4.^2+dt.^2.*((-1)+ubar).*(2.*a3+a2.*((-1)+2.* ...
  ubar)).*((-1)+ubarp).*(2.*a3+a2.*((-1)+2.*ubarp))+2.*a4.*dt.*(2.*a3.*(( ...
  -2)+ubar+ubarp)+a2.*(2+(-3).*ubar+2.*ubar.^2+(-3).*ubarp+2.*ubarp.^2)))+ ...
  2.*a1.*dt.*exp(1).^(2.*logtheta).*(4.*a4+2.*a3.*dt.*((-2)+ubar+ubarp)+ ...
  a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3).*ubarp+2.*ubarp.^2)).*(exp(1) ...
  .^logtheta+(-1).*(x+(-1).*xp).^2)+4.*a1.^2.*dt.^2.*(3.*exp(1).^(2.* ...
  logtheta)+(-6).*exp(1).^logtheta.*(x+(-1).*xp).^2+(x+(-1).*xp).^4)).*(( ...
  -4)+(1/2).*exp(1).^((-1).*logtheta).*(x+(-1).*xp).^2));


case 3 % a1

K=(1/2).*a4.^(-2).*dt.*exp(1).^(logsigma+(-4).*logtheta+(-1/2).*exp(1).^(( ...
  -1).*logtheta).*(x+(-1).*xp).^2).*(exp(1).^(2.*logtheta).*(4.*a4+2.*a3.* ...
  dt.*((-2)+ubar+ubarp)+a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3).*ubarp+2.* ...
  ubarp.^2)).*(exp(1).^logtheta+(-1).*(x+(-1).*xp).^2)+4.*a1.*dt.*(3.*exp( ...
  1).^(2.*logtheta)+(-6).*exp(1).^logtheta.*(x+(-1).*xp).^2+(x+(-1).*xp) ...
  .^4));


case 4 % a2

K=(1/4).*a4.^(-2).*dt.*exp(1).^(logsigma+(-2).*logtheta+(-1/2).*exp(1).^(( ...
  -1).*logtheta).*(x+(-1).*xp).^2).*(exp(1).^(2.*logtheta).*(a4.*(4+(-6).* ...
  ubar+4.*ubar.^2+(-6).*ubarp+4.*ubarp.^2)+2.*dt.*((-1)+ubar).*((-1)+ ...
  ubarp).*(2.*a3.*((-1)+ubar+ubarp)+a2.*((-1)+2.*ubar).*((-1)+2.*ubarp)))+ ...
  2.*a1.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3).*ubarp+2.*ubarp.^2).*(exp(1) ...
  .^logtheta+(-1).*(x+(-1).*xp).^2));


case 5 % a3

K=(1/2).*a4.^(-2).*dt.*exp(1).^(logsigma+(-2).*logtheta+(-1/2).*exp(1).^(( ...
  -1).*logtheta).*(x+(-1).*xp).^2).*(exp(1).^(2.*logtheta).*(2.*a4.*((-2)+ ...
  ubar+ubarp)+2.*dt.*((-1)+ubar).*((-1)+ubarp).*(2.*a3+a2.*((-1)+ubar+ ...
  ubarp)))+2.*a1.*dt.*((-2)+ubar+ubarp).*(exp(1).^logtheta+(-1).*(x+(-1).* ...
  xp).^2));


case 6 % a4

K=(1/4).*a4.^(-3).*exp(1).^(logsigma+(-4).*logtheta+(-1/2).*exp(1).^((-1) ...
  .*logtheta).*(x+(-1).*xp).^2).*(a4.*exp(1).^(4.*logtheta).*(8.*a4+4.* ...
  a3.*dt.*((-2)+ubar+ubarp)+2.*a2.*dt.*(2+(-3).*ubar+2.*ubar.^2+(-3).* ...
  ubarp+2.*ubarp.^2))+(-2).*exp(1).^(4.*logtheta).*(4.*a4.^2+dt.^2.*((-1)+ ...
  ubar).*(2.*a3+a2.*((-1)+2.*ubar)).*((-1)+ubarp).*(2.*a3+a2.*((-1)+2.* ...
  ubarp))+2.*a4.*dt.*(2.*a3.*((-2)+ubar+ubarp)+a2.*(2+(-3).*ubar+2.* ...
  ubar.^2+(-3).*ubarp+2.*ubarp.^2)))+8.*a1.*a4.*dt.*exp(1).^(2.*logtheta) ...
  .*(exp(1).^logtheta+(-1).*(x+(-1).*xp).^2)+(-4).*a1.*dt.*exp(1).^(2.* ...
  logtheta).*(4.*a4+2.*a3.*dt.*((-2)+ubar+ubarp)+a2.*dt.*(2+(-3).*ubar+2.* ...
  ubar.^2+(-3).*ubarp+2.*ubarp.^2)).*(exp(1).^logtheta+(-1).*(x+(-1).*xp) ...
  .^2)+(-8).*a1.^2.*dt.^2.*(3.*exp(1).^(2.*logtheta)+(-6).*exp(1) ...
  .^logtheta.*(x+(-1).*xp).^2+(x+(-1).*xp).^4));


otherwise
        
        K = zeros(n_x, n_xp);
end

if K == 0

    K = zeros(n_x, n_xp);

end

end
