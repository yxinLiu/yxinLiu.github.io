console.log('非源码，仅用作演示。下载源码请访问：HTTP://www.bootstrapmb.com');console.log('非源码，仅用作演示。下载源码请访问：HTTP://WWW.BOOTSTRAPMb.com');console.log('非源码，仅用作演示。下载源码请访问：HTTP://WWW.BOOTStrapmb.com');console.log('非源码，仅用作演示。下载源码请访问：HTtp://www.bootstrapmb.com');(function(a){var c=a.scrollTo=function(d,e,f){a(window).scrollTo(d,e,f)};c.defaults={axis:"xy",duration:parseFloat(a.fn.jquery)>=1.3?0:1,limit:true};c.window=function(d){return a(window)._scrollable()};a.fn._scrollable=function(){return this.map(function(){var d=this,f=!d.nodeName||a.inArray(d.nodeName.toLowerCase(),["iframe","#document","html","body"])!=-1;if(!f){return d}var e=(d.contentWindow||d).document||d.ownerDocument||d;return/webkit/i.test(navigator.userAgent)||e.compatMode=="BackCompat"?e.body:e.documentElement})};a.fn.scrollTo=function(d,e,f){if(typeof e=="object"){f=e;e=0}if(typeof f=="function"){f={onAfter:f}}if(d=="max"){d=9000000000}f=a.extend({},c.defaults,f);e=e||f.duration;f.queue=f.queue&&f.axis.length>1;if(f.queue){e/=2}f.offset=b(f.offset);f.over=b(f.over);return this._scrollable().each(function(){if(!d){return}var l=this,g=a(l),m=d,s,k={},t=g.is("html,body");switch(typeof m){case"number":case"string":if(/^([+-]=)?\d+(\.\d+)?(px|%)?$/.test(m)){m=b(m);break}m=a(m,this);if(!m.length){return}case"object":if(m.is||m.style){s=(m=a(m)).offset()}}a.each(f.axis.split(""),function(o,h){var i=h=="x"?"Left":"Top",u=i.toLowerCase(),p="scroll"+i,r=l[p],q=c.max(l,h);if(s){k[p]=s[u]+(t?0:r-g.offset()[u]);if(f.margin){k[p]-=parseInt(m.css("margin"+i))||0;k[p]-=parseInt(m.css("border"+i+"Width"))||0}k[p]+=f.offset[u]||0;if(f.over[u]){k[p]+=m[h=="x"?"width":"height"]()*f.over[u]}}else{var n=m[u];k[p]=n.slice&&n.slice(-1)=="%"?parseFloat(n)/100*q:n}if(f.limit&&/^\d+$/.test(k[p])){k[p]=k[p]<=0?0:Math.min(k[p],q)}if(!o&&f.queue){if(r!=k[p]){j(f.onAfterFirst)}delete k[p]}});j(f.onAfter);function j(h){g.animate(k,e,f.easing,h&&function(){h.call(this,d,f)})}}).end()};c.max=function(d,e){var g=e=="x"?"Width":"Height",j="scroll"+g;if(!a(d).is("html,body")){return d[j]-a(d)[g.toLowerCase()]()}var h="client"+g,i=d.ownerDocument.documentElement,f=d.ownerDocument.body;return Math.max(i[j],f[j])-Math.min(i[h],f[h])};function b(d){return typeof d=="object"?d:{top:d,left:d}}})(jQuery);console.log('非源码，仅用作演示。下载源码请访问：HTtp://www.bootstrapmb.com');console.log('非源码，仅用作演示。下载源码请访问：HTTP://WWW.BOOTStrapmb.com');console.log('非源码，仅用作演示。下载源码请访问：HTTP://WWW.BOOTSTRAPMb.com');console.log('非源码，仅用作演示。下载源码请访问：HTTP://www.bootstrapmb.com');